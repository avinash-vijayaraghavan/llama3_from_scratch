/* Training for Llama-2 Transformer model in pure C */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdint.h>
#include <assert.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#ifdef OMP
#include <omp.h>
#endif
#if defined _WIN32
#include "win.h"
#else
#include <unistd.h>
#include <sys/mman.h>
#endif

#define DEBUG
// ----------------------------------------------------------------------------
// Transformer model

// similar to ModelArgs (in model.py)
typedef struct
{
    int seq_len;     // max sequence length
    int vocab_size;  // vocabulary size, usually 256 (byte-level)
    int n_layers;    // number of layers
    int n_heads;     // number of query heads
    int n_kv_heads;  // number of key/value heads (can be < query heads because of multiquery)
    int dim;         // transformer dimension
    int hidden_dim;  // for ffn layers
    int multiple_of; // 1024 ?
    float ffn_dim_multiplier;
    float norm_eps;
    float rope_theta;
} Config;

// weights for all the layers in the transformer. this will be created
// and copied to a contiguous block of memory
typedef struct
{
    // token embedding table
    int shared_weights;
    float *token_embedding_table; // (vocab_size, dim)

    // weights for rmsnorms
    float *rms_att_weight; // (layer, dim) rmsnorm weights
    float *rms_ffn_weight; // (layer, dim)

    // weights for matmuls. note dim == n_heads * head_size
    float *wq; // (layer, dim, n_heads * head_size)
    float *wk; // (layer, dim, n_kv_heads * head_size)
    float *wv; // (layer, dim, n_kv_heads * head_size)
    float *wo; // (layer, n_heads * head_size, dim)

    // weights for ffn
    float *w1; // (layer, hidden_dim, dim)
    float *w2; // (layer, dim, hidden_dim)
    float *w3; // (layer, hidden_dim, dim)

    // final rmsnorm
    float *rms_final_weight; // (dim,)

    // (optional) classifier weights for the logits, on the last layer
    float *wcls; // dim, vocab_size
} TransformerWeights;

typedef struct
{
    // current activations
    float *x;               // activation at current time stamp (B, T ,dim,)
    float *layer_input;     // input to the next layer
    float *x_rms_attn_norm; // after rms norm and before attention (L, B,T,dim,)
    float *q;               // query (L,B,T,dim,)  q,k,v
    float *k;               // key (L,B,T,kv_dim,)
    float *v;               // value (L,B,T, kv_dim,)

    // rope
    float *qr; // query (L,B,T,dim,)  q,k,v
    float *kr; // key (L,B,T,kv_dim,)

    float *att;                // buffer for attention values (L, B, n_heads, seq_len, seq_len)
    float *preatt;             // buffer for scores (q.K_T) values (L, B, n_heads, seq_len, seq_len)
    float *attn_out;           // L,B,T,C
    float *x_attn;             // after self-attention, before rms_norm of ffn
    float *xo;                 // proj of attn_out into residual atten_out * wo
    float *x_res;              // x + residual,
    float *x_res_rms_ffn_norm; // rms norm of x_res

    // ffn
    float *h1;         // buffer for hidden dimension (*w1)in the ffn (L,B,T,OC,)
    float *h2;         // buffer for hidden dimension in(*w3) the ffn (L,B,T,C,)
    float *h3;         // buffer for hidden dimension (*w1)in the ffn (L,B,T,OC,)
    float *h1_h3_prod; // buffer for hidden dimension in(*w3) the ffn (L,B,T,OC,)
    float *h1_silu;    // buffer for hidden dimension in(*w3) the ffn (L,B,T,OC,)

    float *x_ffn;       // after matmul in feedforeard (*w2)
    float *x_final_res; // final residual (x_attn + x_ffn), this is the input to the next layer

    float *x_final_rms_norm; // final rms norm begore linear layer

    float *logits; // output logits (B, T, vocab_size)
    float *prob;   // (B, T, vocab_size)
    float *loss;   // B,T
} TrainState;      // train state will need to store more activations (to enable backprop)

void free_weights(TransformerWeights *w)
{
    free(w->token_embedding_table);

    free(w->rms_att_weight);
    free(w->rms_ffn_weight);

    free(w->wq);
    free(w->wk);
    free(w->wv);
    free(w->wo);

    free(w->w1);
    free(w->w2);
    free(w->w3);

    free(w->rms_final_weight);
}
typedef struct
{
    Config config;               // the hyperparameters of the architecture (the blueprint)
    TrainState train_state;      // acts
    TrainState grad_train_state; // grad_acts
    TransformerWeights weights;  // the weights of the model
    TransformerWeights dweights; // grad of the transformer weights

    // addtional buffers for the adam optimzer
    TransformerWeights m_dweights;
    TransformerWeights var_dweights;

    int fd;            // file descriptor for memory mapping
    float *data;       // memory mapped data pointer
    ssize_t file_size; // size of the checkpoint file in bytes (the chkpt file is mapped to memory to create the transformer weights)
    int batch_size;
    int seq_len; // seq len for the current batch, MUST be <= config.seq_len
    int *inputs; // input tokens
    int *targets;
    float mean_loss;
    int num_params;
} Transformer;

void malloc_train_state(TrainState *ts, Config *p, int B, int T)
{
#define EXIT(x)                         \
    {                                   \
        printf("%s malloc error\n", x); \
        exit(EXIT_FAILURE);             \
    }
    // include the grads also

    assert(T <= p->seq_len);
    printf("Allocating train state\n");
    int L = p->n_layers;
    int head_size = p->dim / p->n_heads;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads; // attention
    int n_heads = p->n_heads;
    ts->x = (float *)calloc(B * T * p->dim, sizeof(float)); // x , grad x
    if (!ts->x)
        EXIT("x")
    ts->x_rms_attn_norm = (float *)calloc(L * B * T * p->dim, sizeof(float)); // xb , grad xb
    if (!ts->x_rms_attn_norm)
        EXIT("x_rms_attn_norm")

    ts->q = (float *)calloc(L * B * T * p->dim, sizeof(float)); // q , grad q
    if (!ts->q)
        EXIT("q")
    ts->k = (float *)calloc(L * B * T * kv_dim, sizeof(float)); // k , grad k
    if (!ts->k)
        EXIT("k")
    ts->v = (float *)calloc(L * B * T * kv_dim, sizeof(float)); // v , grad v
    if (!ts->v)
        EXIT("v")
    // rope
    ts->qr = (float *)calloc(L * B * T * p->dim, sizeof(float)); // q , grad q
    if (!ts->qr)
        EXIT("qr")
    ts->kr = (float *)calloc(L * B * T * kv_dim, sizeof(float)); // k , grad k
    if (!ts->kr)
        EXIT("kr")
    ts->x_attn = (float *)calloc(L * B * T * p->dim, sizeof(float));
    if (!ts->x_attn)
        EXIT("x_attn")
    ts->attn_out = (float *)calloc(L * B * T * p->dim, sizeof(float));
    if (!ts->attn_out)
        EXIT("attn_out")
    ts->xo = (float *)calloc(L * B * T * p->dim, sizeof(float));
    if (!ts->xo)
        EXIT("xo")
    ts->x_res = (float *)calloc(L * B * T * p->dim, sizeof(float)); // hb , grad hb
    if (!ts->x_res)
        EXIT("x_res")
    ts->x_res_rms_ffn_norm = (float *)calloc(L * B * T * p->dim, sizeof(float)); // hb , grad hb
    if (!ts->x_res_rms_ffn_norm)
        EXIT("x_res_rms_ffn_norm")
    ts->h1 = (float *)calloc(L * B * T * p->hidden_dim, sizeof(float)); // hb2 , grad hb2
    if (!ts->h1)
        EXIT("h1")
    ts->h3 = (float *)calloc(L * B * T * p->hidden_dim, sizeof(float)); // hb2 , grad hb2
    if (!ts->h3)
        EXIT("h3")
    ts->h1_silu = (float *)calloc(L * B * T * p->hidden_dim, sizeof(float)); // hb2 , grad hb2
    if (!ts->h1_silu)
        EXIT("h1_silu")
    ts->h1_h3_prod = (float *)calloc(L * B * T * p->hidden_dim, sizeof(float)); // hb2 , grad hb2
    if (!ts->h1_h3_prod)
        EXIT("h1_h3_prod")
    ts->h2 = (float *)calloc(L * B * T * p->dim, sizeof(float)); // hb2 , grad hb2
    if (!ts->h2)
        EXIT("h2")
    ts->x_ffn = (float *)calloc(L * B * T * p->dim, sizeof(float)); // hb2 , grad hb2
    if (!ts->x_ffn)
        EXIT("x_ffn")
    ts->x_final_res = (float *)calloc(L * B * T * p->dim, sizeof(float)); // hb2 , grad hb2
    if (!ts->x_final_res)
        EXIT("x_final_res")
    ts->x_final_rms_norm = (float *)calloc(B * T * p->dim, sizeof(float));
    if (!ts->x_final_rms_norm)
        EXIT("x_final_rms_norm")
    ts->att = (float *)calloc(L * B * n_heads * T * T, sizeof(float)); // att. grad att
    if (!ts->att)
        EXIT("att")
    ts->preatt = (float *)calloc(L * B * n_heads * T * T, sizeof(float)); // att. grad att
    if (!ts->preatt)
        EXIT("preatt")
    ts->logits = (float *)calloc(B * T * p->vocab_size, sizeof(float));
    if (!ts->logits)
        EXIT("logits")
    ts->prob = (float *)calloc(B * T * p->vocab_size, sizeof(float));
    if (!ts->prob)
        EXIT("prob")
    ts->loss = (float *)calloc(B * T, sizeof(float));
    if (!ts->loss)
        EXIT("loss")
    // ensure all mallocs went fine
    if (!ts->x || !ts->x_rms_attn_norm || !ts->q || !ts->k || !ts->v || !ts->qr || !ts->kr ||
        !ts->attn_out || !ts->xo || !ts->x_res || !ts->x_res_rms_ffn_norm || !ts->h1 || !ts->h2 ||
        !ts->h3 || !ts->h1_silu || !ts->h1_h3_prod || !ts->x_ffn || !ts->x_ffn || !ts->x_final_res ||
        !ts->logits || !ts->prob || !ts->loss || !ts->att || !ts->preatt)

    {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }
}

void free_train_state(TrainState *ts)
{
    free(ts->x);
    free(ts->x_rms_attn_norm);
    free(ts->x_final_rms_norm);
    free(ts->q);
    free(ts->k);
    free(ts->v);
    free(ts->qr);
    free(ts->kr);

    free(ts->att);
    free(ts->preatt);
    free(ts->x_attn);
    free(ts->attn_out);
    free(ts->xo); // ok
    free(ts->x_res);
    free(ts->x_res_rms_ffn_norm);

    free(ts->h1);
    free(ts->h2);
    free(ts->h3);
    free(ts->h1_h3_prod);
    free(ts->h1_silu);

    free(ts->x_ffn);
    free(ts->x_final_res);

    free(ts->logits);
    free(ts->loss);
    free(ts->prob);
}

void adam_update_weights(float *w, float *dw, float *m_dw, float *var_dw, int N, float learning_rate, float beta1, float beta2, float eps, float weight_decay, int iter)
{
    // adam update
    float m, v, m_hat, v_hat;
    for (int i = 0; i < N; i++)
    {
        m = beta1 * m_dw[i] + (1 - beta1) * dw[i];
        v = beta2 * var_dw[i] + (1 - beta2) * dw[i] * dw[i];
        m_hat = m / (1.0f - powf(beta1, iter));
        v_hat = v / (1.0f - powf(beta2, iter));

        m_dw[i] = m;
        var_dw[i] = v;
        w[i] -= learning_rate * (m_hat / (1.0 * sqrt(v_hat) + eps) + weight_decay * w[i]);
        // printf("i: %d, m: %f, v: %f, w[%d]: %f\n", i, m, v,i, w[i]);
    }
}
void update_weights(Transformer *model, float learning_rate, float beta1, float beta2, float eps, float weight_decay, int iter)
{
    Config c = model->config;
    int L = c.n_layers;
    int C = c.dim;
    int nh = c.n_heads;
    int n_kv_heads = c.n_kv_heads;
    int OC = c.hidden_dim;
    int V = c.vocab_size;
    int head_size = C / nh;

    TransformerWeights *dw = &model->dweights;
    TransformerWeights *w = &model->weights;
    TransformerWeights *m_dw = &model->m_dweights;
    TransformerWeights *var_dw = &model->var_dweights;

    float *wte = w->token_embedding_table;
    float *dwte = dw->token_embedding_table;
    float *m_dwte = m_dw->token_embedding_table;
    float *var_dwte = var_dw->token_embedding_table;
    adam_update_weights(wte, dwte, m_dwte, var_dwte, V * C, learning_rate, beta1, beta2, eps, weight_decay, iter);
    printf("wte updated\n");
    adam_update_weights(w->rms_att_weight, dw->rms_att_weight, m_dw->rms_att_weight, var_dw->rms_att_weight, L * C, learning_rate, beta1, beta2, eps, weight_decay, iter);
    printf("rms_att_weight updated\n");
    adam_update_weights(w->rms_ffn_weight, dw->rms_ffn_weight, m_dw->rms_ffn_weight, var_dw->rms_ffn_weight, L * C, learning_rate, beta1, beta2, eps, weight_decay, iter);
    printf("rms_ffn_weight updated\n");
    adam_update_weights(w->wq, dw->wq, m_dw->wq, var_dw->wq, L * C * C, learning_rate, beta1, beta2, eps, weight_decay, iter);
    printf("wq updated\n");
    adam_update_weights(w->wo, dw->wo, m_dw->wo, var_dw->wo, L * C * C, learning_rate, beta1, beta2, eps, weight_decay, iter);
    printf("wo updated\n");
    adam_update_weights(w->wk, dw->wk, m_dw->wk, var_dw->wk, L * C * n_kv_heads * head_size, learning_rate, beta1, beta2, eps, weight_decay, iter);
    printf("wk updated\n");
    adam_update_weights(w->wv, dw->wv, m_dw->wv, var_dw->wv, L * C * n_kv_heads * head_size, learning_rate, beta1, beta2, eps, weight_decay, iter);
    printf("wv updated\n");
    adam_update_weights(w->w1, dw->w1, m_dw->w1, var_dw->w1, L * C * OC, learning_rate, beta1, beta2, eps, weight_decay, iter);
    printf("w1 updated\n");
    adam_update_weights(w->w2, dw->w2, m_dw->w2, var_dw->w2, L * C * OC, learning_rate, beta1, beta2, eps, weight_decay, iter);
    printf("w2 updated\n");
    adam_update_weights(w->w3, dw->w3, m_dw->w3, var_dw->w3, L * C * OC, learning_rate, beta1, beta2, eps, weight_decay, iter);
    printf("w3 updated\n");
    adam_update_weights(w->rms_final_weight, dw->rms_final_weight, m_dw->rms_final_weight, var_dw->rms_final_weight, C, learning_rate, beta1, beta2, eps, weight_decay, iter);
    printf("rms_final_weight updated\n");
}

// ptr is the weights pointer in the memory mapped region of the checkpoint file
int memory_map_weights(TransformerWeights *w, Config *p, float *ptr, int shared_weights)
{
    int params = 0;
    int head_size = p->dim / p->n_heads;
    // make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ models
    unsigned long long n_layers = p->n_layers;
    w->token_embedding_table = ptr; // encoding table (vacab_size , dim)
    ptr += p->vocab_size * p->dim;
    params += 1; // p->vocab_size * p->dim;

    w->wcls = ptr;
    ptr += p->vocab_size * p->dim;
    params += 1;

    w->rms_att_weight = ptr; // (layer, dim) rmsnorm weights
    ptr += n_layers * p->dim;
    params += n_layers; // * p->dim;

    w->wq = ptr; // wq, wo: (layers * dim * head_dim * n_heads)
    ptr += n_layers * p->dim * (p->dim);
    // params += n_layers; // * p->dim * (p->n_heads * head_size);

    w->wk = ptr; // wk, wv : (layers * dim * head_dim * n_kv_heads)
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    // params += n_layers; // * p->dim * (p->n_kv_heads * head_size);

    w->wv = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    params += n_layers; // * p->dim * (p->n_kv_heads * head_size);

    w->wo = ptr;
    ptr += n_layers * (p->dim) * p->dim;
    params += n_layers; // * (p->n_heads * head_size) * p->dim;

    w->rms_ffn_weight = ptr; // (layer, dim) rmsnorm weights
    ptr += n_layers * p->dim;
    params += n_layers; // * p->dim;

    w->w1 = ptr; // w1, w3: (layers * dim * hidden_dim)
    ptr += n_layers * p->dim * p->hidden_dim;
    params += n_layers; // * p->dim * p->hidden_dim;

    w->w3 = ptr; // w2: (layers * dim * hidden_dim)
    ptr += n_layers * p->hidden_dim * p->dim;
    params += n_layers; // * p->hidden_dim * p->dim;

    w->w2 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    params += n_layers; // * p->dim * p->hidden_dim;

    w->rms_final_weight = ptr;
    ptr += p->dim;
    params += 1; // p->dim;

    // printf("Shared weights: %d\n", shared_weights);
    printf("Total number of params: %ld\n", params);
    return params;
}
void malloc_grad_weights(Transformer *model)
{

    TransformerWeights *dw = &model->dweights;
    TransformerWeights *m_dw = &model->m_dweights;
    TransformerWeights *var_dw = &model->var_dweights;
    Config *p = &model->config;
    int head_size = p->dim / p->n_heads;
    // allocate the memory for the grad weights
    int shared_weights = 0;

    dw->token_embedding_table = (float *)calloc(p->vocab_size * p->dim, sizeof(float));     // (vocab_size, dim)
    m_dw->token_embedding_table = (float *)calloc(p->vocab_size * p->dim, sizeof(float));   // (vocab_size, dim)
    var_dw->token_embedding_table = (float *)calloc(p->vocab_size * p->dim, sizeof(float)); // (vocab_size, dim)

    // weights for rmsnorms
    dw->rms_att_weight = (float *)calloc(p->n_layers * p->dim, sizeof(float));     // (layer, dim) rmsnorm weights
    m_dw->rms_att_weight = (float *)calloc(p->n_layers * p->dim, sizeof(float));   // (layer, dim) rmsnorm weights
    var_dw->rms_att_weight = (float *)calloc(p->n_layers * p->dim, sizeof(float)); // (layer, dim) rmsnorm weights

    dw->rms_ffn_weight = (float *)calloc(p->n_layers * p->dim, sizeof(float));     // (layer, dim)
    m_dw->rms_ffn_weight = (float *)calloc(p->n_layers * p->dim, sizeof(float));   // (layer, dim)
    var_dw->rms_ffn_weight = (float *)calloc(p->n_layers * p->dim, sizeof(float)); // (layer, dim)

    // weights for matmuls. note dim == n_heads * head_size

    dw->wq = (float *)calloc(p->n_layers * p->dim * p->n_heads * head_size, sizeof(float));     // (layer, dim, n_heads * head_size)
    m_dw->wq = (float *)calloc(p->n_layers * p->dim * p->n_heads * head_size, sizeof(float));   // (layer, dim, n_heads * head_size)
    var_dw->wq = (float *)calloc(p->n_layers * p->dim * p->n_heads * head_size, sizeof(float)); // (layer, dim, n_heads * head_size)

    dw->wk = (float *)calloc(p->n_layers * p->dim * p->n_kv_heads * head_size, sizeof(float));     // (layer, dim, n_kv_heads * head_size)
    m_dw->wk = (float *)calloc(p->n_layers * p->dim * p->n_kv_heads * head_size, sizeof(float));   // (layer, dim, n_kv_heads * head_size)
    var_dw->wk = (float *)calloc(p->n_layers * p->dim * p->n_kv_heads * head_size, sizeof(float)); // (layer, dim, n_kv_heads * head_size)

    dw->wv = (float *)calloc(p->n_layers * p->dim * p->n_kv_heads * head_size, sizeof(float));     // (layer, dim, n_kv_heads * head_size)
    m_dw->wv = (float *)calloc(p->n_layers * p->dim * p->n_kv_heads * head_size, sizeof(float));   // (layer, dim, n_kv_heads * head_size)
    var_dw->wv = (float *)calloc(p->n_layers * p->dim * p->n_kv_heads * head_size, sizeof(float)); // (layer, dim, n_kv_heads * head_size)

    dw->wo = (float *)calloc(p->n_layers * p->dim * p->n_heads * head_size, sizeof(float));     // (layer, n_heads * head_size, dim)
    m_dw->wo = (float *)calloc(p->n_layers * p->dim * p->n_heads * head_size, sizeof(float));   // (layer, n_heads * head_size, dim)
    var_dw->wo = (float *)calloc(p->n_layers * p->dim * p->n_heads * head_size, sizeof(float)); // (layer, n_heads * head_size, dim)

    // weights for ffn
    dw->w1 = (float *)calloc(p->n_layers * p->hidden_dim * p->dim, sizeof(float));     // (layer, hidden_dim, dim)
    m_dw->w1 = (float *)calloc(p->n_layers * p->hidden_dim * p->dim, sizeof(float));   // (layer, hidden_dim, dim)
    var_dw->w1 = (float *)calloc(p->n_layers * p->hidden_dim * p->dim, sizeof(float)); // (layer, hidden_dim, dim)

    dw->w2 = (float *)calloc(p->n_layers * p->hidden_dim * p->dim, sizeof(float));     // (layer, dim, hidden_dim)
    m_dw->w2 = (float *)calloc(p->n_layers * p->hidden_dim * p->dim, sizeof(float));   // (layer, dim, hidden_dim)
    var_dw->w2 = (float *)calloc(p->n_layers * p->hidden_dim * p->dim, sizeof(float)); // (layer, dim, hidden_dim)

    dw->w3 = (float *)calloc(p->n_layers * p->hidden_dim * p->dim, sizeof(float));     // (layer, hidden_dim, dim)
    m_dw->w3 = (float *)calloc(p->n_layers * p->hidden_dim * p->dim, sizeof(float));   // (layer, hidden_dim, dim)
    var_dw->w3 = (float *)calloc(p->n_layers * p->hidden_dim * p->dim, sizeof(float)); // (layer, hidden_dim, dim)

    // final rmsnorm
    dw->rms_final_weight = (float *)calloc(p->dim, sizeof(float));     // (dim,)
    m_dw->rms_final_weight = (float *)calloc(p->dim, sizeof(float));   // (dim,)
    var_dw->rms_final_weight = (float *)calloc(p->dim, sizeof(float)); // (dim,)

    dw->wcls = (float *)calloc(p->dim * p->vocab_size, sizeof(float));
    printf("Done allocation of gradients of weights\n");
}
int read_checkpoint(Transformer *model, char *checkpoint)
// Config *config, TransformerWeights *weights,
// int *fd, float **data, ssize_t *file_size)
{
    FILE *file = fopen(checkpoint, "rb");
    if (!file)
    {
        fprintf(stderr, "Couldn't open file %s\n", checkpoint);
        exit(EXIT_FAILURE);
    }
    // read in the config header
    int header_int[256]; // int section of the header
    ssize_t file_size;
    int fd;
    float *data = (float *)malloc(sizeof(float));
    TransformerWeights *weights = &model->weights;
    fread(header_int, sizeof(int), 256, file);
    assert(sizeof(int) == 4); // i think the python export code currently assumes this is int32
    float header_float[256];  // float section of the header
    fread(header_float, sizeof(float), 256, file);
    int version = header_int[1];
    if (!(version == 3 || version == 5))
    {
        // 3 = fp32, padded vocab
        // 5 = bf16, padded vocab, layernorms also in bf16
        fprintf(stderr, "Bad version in model file\n");
        fprintf(stderr, "---> HINT: try to re-run `python train_gpt2.py`\n");
        exit(EXIT_FAILURE);
    }
    Config *config = &model->config;
    config->seq_len = header_int[2];
    config->vocab_size = header_int[3];
    config->n_layers = header_int[4];
    config->n_heads = header_int[5];
    config->n_kv_heads = header_int[6];
    config->dim = header_int[7];
    config->multiple_of = header_int[8];

    config->ffn_dim_multiplier = header_float[0];
    config->norm_eps = header_float[1];
    config->rope_theta = header_float[2];

    // compute the hidden dim
    int hidden_dim = 4 * config->dim;
    hidden_dim = (2 * hidden_dim) / 3;
    hidden_dim = (config->ffn_dim_multiplier * hidden_dim);
    hidden_dim = config->multiple_of * ((hidden_dim + config->multiple_of - 1) / config->multiple_of);
    config->hidden_dim = hidden_dim;

    // figure out the file size
    fseek(file, 0, SEEK_END); // move file pointer to end of file
    file_size = ftell(file);  // get the file size, in bytes
    fclose(file);
    // memory map the Transformer weights into the data pointer
    fd = open(checkpoint, O_RDONLY); // open in read only mode
    if (fd == -1)
    {
        fprintf(stderr, "open failed!\n");
        exit(EXIT_FAILURE);
    }
    data = mmap(NULL, file_size, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
    if ((void *)data == MAP_FAILED)
    {
        fprintf(stderr, "mmap failed!\n");
        exit(EXIT_FAILURE);
    }
    float *weights_ptr = data + 512; // header_int[256], header_float[256], params
    return memory_map_weights(weights, &model->config, weights_ptr, 0);
}

void build_transformer(Transformer *t, char *checkpoint_path)
{
    // read in the Config and the Weights from the checkpoint
    // t->num_params = read_checkpoint(checkpoint_path, &(t->config), &(t->weights), &(t->fd), &(t->data), &(t->file_size));
    t->num_params = read_checkpoint(t, checkpoint_path);

    // allocate grads of weights
    malloc_grad_weights(t);
}

// void free_transformer(Transformer *t)
// {
//     // close the memory mapping
//     if (t->data != MAP_FAILED)
//     {
//         munmap(t->data, t->file_size);
//     }
//     if (t->fd != -1)
//     {
//         close(t->fd);
//     }
//     // free the RunState buffers
//     free_run_state(&t->state);
// }

void free_trained_transformer(Transformer *t)
{
    // close the memory mapping
    if (t->data != MAP_FAILED)
    {
        munmap(t->data, t->file_size); // for freeing weights
    }
    if (t->fd != -1)
    {
        close(t->fd);
    }
    // free the RunState buffers

    free_train_state(&t->train_state);
    free_train_state(&t->grad_train_state);
    // free_weights(&t->weights);
    free_weights(&t->dweights);
    free_weights(&t->m_dweights);
    free_weights(&t->var_dweights);
}

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

void encoder_forward(float *out, int *in, float *wte, int B, int T, int C)
{
    // in(B,T) -> sentence, token
    // wte(V, C)
    // out(B,T,C)
    for (int b = 0; b < B; b++)
    {
        for (int t = 0; t < T; t++)
        {
            float *out_bt = out + b * T * C + t * C;
            int token = in[b * T + t];
            float *enc = wte + token * C;
            // //printf("%d %f\n",token, enc[0]);
            for (int c = 0; c < C; c++)
                out_bt[c] = enc[c];
        }
    }
}

void encoder_backward(float *dwte,
                      float *dout, int *inp,
                      int B, int T, int C)
{
    for (int b = 0; b < B; b++)
    {
        for (int t = 0; t < T; t++)
        {
            float *dout_bt = dout + b * T * C + t * C;
            int token = inp[b * T + t];
            float *dwte_token = dwte + token * C;
            for (int i = 0; i < C; i++)
            {
                float d = dout_bt[i];
                dwte_token[i] += d;
            }
        }
    }
}

void rmsnorm_forward(float *o, float *x, float *weight, int B, int T, int C)
{
    // calculate sum of squares
    // x (B,T,C)
    // o (B,T,C)
    // weight (C)
    for (int b = 0; b < B; b++)
    {
        for (int t = 0; t < T; t++)
        {
            float ss = 0.0f;
            for (int j = 0; j < C; j++)
            {
                float *in = x + b * T * C + t * C;
                ss += in[j] * in[j];
            }
            ss /= C;
            // //printf("ss: %f\n",ss);
            ss += 1e-5f;
            ss = 1.0f / sqrtf(ss);
            // normalize and scale
            for (int j = 0; j < C; j++)
            {
                float *in = x + b * T * C + t * C;
                float *out = o + b * T * C + t * C;
                out[j] = weight[j] * (ss * in[j]);
            }
        }
    }
}

void rmsnorm_backward(float *dout, float *dx, float *dweight, float *x, float *w, int B, int T, int C)
{
    // dx (B,T,C)
    // dout (B,T,C)
    // dweight (C)
    for (int b = 0; b < B; b++)
    {
        for (int t = 0; t < T; t++)
        {
            float ss = 0.0f;
            float *in = x + b * T * C + t * C;
            float *dout_ix = dout + b * T * C + t * C;
            float *dx_ix = dx + b * T * C + t * C;

            for (int j = 0; j < C; j++)
                ss += in[j] * in[j];

            ss /= C;
            ss += 1e-5f;
            ss = 1.0f / sqrtf(ss);
            for (int j = 0; j < C; j++)
            {
                dweight[j] += dout_ix[j] * in[j] * ss;
                for (int k = 0; k < C; k++)
                    dx_ix[j] += j == k ? dout_ix[j] * w[j] * (ss - powf(in[j], 2.0) * powf(ss, 3.0) / C) : -dout_ix[k] * w[k] * in[k] * in[j] * powf(ss, 3.0) / C;
            }
        }
    }
    sum_vector(dx, B * T * C, -1, 5, "rmsnorm_dx");
    sum_vector(dweight, C, -1, 5, "rmsnorm_dweight");
}

void matmul_forward_naive(float *out,
                          const float *inp, const float *weight, const float *bias,
                          int B, int T, int C, int OC)
{
    // the most naive implementation of matrix multiplication
    // this serves as an algorithmic reference, and as a fallback for
    // unfriendly input shapes inside matmul_forward(), below.
    // inp (BTC) weight (C,OC) -> out (BT,OC)
    // #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++)
    {
        for (int t = 0; t < T; t++)
        {
            int bt = b * T + t;
            for (int o = 0; o < OC; o++)
            {
                float val = (bias != NULL) ? bias[o] : 0.0f;
                for (int i = 0; i < C; i++)
                {
                    val += inp[bt * C + i] * weight[o * C + i];
                }
                out[bt * OC + o] = val;
            }
        }
    }
}

void matmul_backward(float *dinp, float *dweight, float *dbias,
                     const float *dout, const float *inp, const float *weight,
                     int B, int T, int C, int OC)
{
    // most of the running time is spent here and in matmul_forward
    // this backward could be done in a single "round" of loops
    // but that doesn't afford an efficient parallelization strategy

    // backward into inp first, parallelize over B,T
    // #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++)
    {
        for (int t = 0; t < T; t++)
        {
            const float *dout_bt = dout + b * T * OC + t * OC;
            float *dinp_bt = dinp + b * T * C + t * C;
            for (int o = 0; o < OC; o++)
            {
                const float *wrow = weight + o * C;
                float d = dout_bt[o];
                for (int i = 0; i < C; i++)
                {
                    dinp_bt[i] += wrow[i] * d;
                }
            }
        }
    }
    // backward into weight/bias, parallelize over output channels OC
    // #pragma omp parallel for
    for (int o = 0; o < OC; o++)
    {
        for (int b = 0; b < B; b++)
        {
            for (int t = 0; t < T; t++)
            {
                const float *dout_bt = dout + b * T * OC + t * OC;
                const float *inp_bt = inp + b * T * C + t * C;
                float *dwrow = dweight + o * C;
                float d = dout_bt[o];
                if (dbias != NULL)
                {
                    dbias[o] += d;
                }
                for (int i = 0; i < C; i++)
                {
                    dwrow[i] += inp_bt[i] * d;
                }
            }
        }
    }
}

void silu(float *out,
          const float *inp, const int B, const int T, const int C)
{
    for (int b = 0; b < B; b++)
        for (int t = 0; t < T; t++)
            for (int c = 0; c < C; c++)
            {
                float x = inp[b * T * C + t * C + c];
                out[b * T * C + t * C + c] = x / (1.0 + expf(-x));
            }
}

void silu_backward(float *dout,
                   float *dinp, float *inp, const int B, const int T, const int C)
{
    for (int b = 0; b < B; b++)
        for (int t = 0; t < T; t++)
            for (int c = 0; c < C; c++)
            {
                float x = inp[b * T * C + t * C + c];
                float expx = expf(-x);
                dinp[b * T * C + t * C + c] = dout[b * T * C + t * C + c] * (1 / (1.0 + expx) + x * expx * powf(1.0 + expx, -2.0));
            }
}

void prod(float *out, float *x1, float *x2, int B, int T, int C)
{
    for (int b = 0; b < B; b++)
        for (int t = 0; t < T; t++)
            for (int c = 0; c < C; c++)
            {
                float in1 = x1[b * T * C + t * C + c];
                float in2 = x2[b * T * C + t * C + c];
                out[b * T * C + t * C + c] = in1 * in2;
            }
}

void feedforward(float *out, float *h1, float *h2, float *h3,
                 float *h1_silu, float *h1_h3_prod,
                 float *inp, float *w1, float *w2, float *w3, const int B, const int T, const int C, const int OC)
{
    // w1, w3 (C, OC)
    // w2 (OC,C)
    // inp (BTC)
    // out (BTC)
    // //printf("%d %d %d %d", B, T, C, OC);

    matmul_forward_naive((float *)h1,
                         (const float *)(inp), (const float *)w1, NULL,
                         B, T, C, OC); // w1(x)
    silu(h1_silu, h1, B, T, OC);       // F.silu(w1(x))
    matmul_forward_naive(h3,
                         (const float *)inp, (const float *)w3, NULL,
                         B, T, C, OC);       // w3(x)
    prod(h1_h3_prod, h1_silu, h3, B, T, OC); // F.silu(w1(x)) * w3(x)
    // //printf("h1_h3_prod: %f %f %f %f\n", h1_h3_prod[0], h1_h3_prod[1], h1_h3_prod[2], h1_h3_prod[3]);
    // //printf("h1_h3_prod: %f %f %f %f\n", h1_h3_prod[OC - 1], h1_h3_prod[OC - 2], h1_h3_prod[OC - 3], h1_h3_prod[OC - 4]);
    matmul_forward_naive((float *)out,
                         (const float *)h1_h3_prod, (const float *)w2, NULL,
                         B, T, OC, C); // w2(temp1)
}

void feedforward_backward(float *dout, float *dh1, float *dh2, float *dh1_silu, float *dh1_h3_prod, float *dh3,
                          float *h1, float *h3, float *h1_silu, float *h1_h3_prod,
                          const float *dinp, float *dw1, float *dw2, float *dw3, float *in, float *out, float *w1, float *w2, float *w3, const int B, const int T, const int C, const int OC)
{
    // in (B,T,C)
    // out -> out, temp_w1, temp_silu, temp_w3, prod
    // dout -> B,T,C
    // din -> B,T,C

    // dout -> d_h1_h3_prod -> dtemp_w3 -> dtemp_silu -> dtemp->w1
    //                                          |-> din

    // float *temp_offset = out + B * T * C;
    // float *t1 = temp_offset;          // w1(x)
    // float *t1_silu = t1 + B * T * OC; // F.silu(w1(x))
    // float *t3 = t1_silu + B * T * OC; // w3(x)
    // float *t1_t3 = t3 + B * T * OC;   // F.silu(w1(x)) *  w3(x)

    // out = w2(F.silu(w1(x)) *  w3(x))
    // float *dt1 = calloc(B * T * OC, sizeof(float));
    // float *dt1_silu = calloc(B * T * OC, sizeof(float));
    // float *dt3 = calloc(B * T * OC, sizeof(float));
    // float *dt1_t3 = calloc(B * T * OC, sizeof(float));

    matmul_backward(dh1_h3_prod, dw2, NULL, dout, h1_h3_prod, w2, B, T, OC, C);
    sum_vector(dh1_h3_prod, B * T * OC, -1, 5, "dh1_h3_prod");
    sum_vector(dw2, OC * C, -1, 5, "dw2");
    for (int b = 0; b < B; b++)
        for (int t = 0; t < T; t++)
            for (int oc = 0; oc < OC; oc++)
            {
                dh1_silu[b * T * OC + t * OC + oc] = dh1_h3_prod[b * T * OC + t * OC + oc] * h3[b * T * OC + t * OC + oc];
                dh3[b * T * OC + t * OC + oc] = dh1_h3_prod[b * T * OC + t * OC + oc] * h1_silu[b * T * OC + t * OC + oc];
            }
    sum_vector(dh1_silu, B * T * OC, -1, 5, "dh1_silu");
    matmul_backward(dinp, dw3, NULL, dh3, in, w3, B, T, C, OC);
    sum_vector(dh3, B * T * OC, -1, 5, "dh3");
    sum_vector(dw3, C * OC, -1, 5, "dw3");
    silu_backward(dh1_silu, dh1, h1, B, T, OC);
    sum_vector(dh1, B * T * OC, -1, 5, "dh1");
    matmul_backward(dinp, dw1, NULL, dh1, in, w1, B, T, C, OC);
    sum_vector(dw1, C * OC, -1, 5, "dw1");
}
void softmax_forward(float *probs, float *logits, int B, int T, int V, int Vp)
{
// output: probs are (B,T,Vp) of the probabilities (sums to 1.0 in each b,t position)
// input: logits is (B,T,Vp) of the unnormalized log probabilities
// Vp is the padded vocab size (for efficiency), V is the "real" vocab size
// example: Vp is 50304 and V is 50257
#pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++)
    {
        for (int t = 0; t < T; t++)
        {
            // probs <- softmax(logits)
            float *logits_bt = logits + b * T * Vp + t * Vp;
            float *probs_bt = probs + b * T * Vp + t * Vp;

            // maxval is only calculated and subtracted for numerical stability
            float maxval = -1000000.0f; // TODO something better
            for (int i = 0; i < V; i++)
            {
                if (logits_bt[i] > maxval)
                {
                    maxval = logits_bt[i];
                }
            }
            float sum = 0.0f;
            for (int i = 0; i < V; i++)
            {
                probs_bt[i] = expf(logits_bt[i] - maxval);
                sum += probs_bt[i];
            }
            // note we only loop to V, leaving the padded dimensions
            for (int i = 0; i < V; i++)
            {
                probs_bt[i] /= sum;
            }
            // for extra super safety we may wish to include this too,
            // forcing the probabilities here to be zero, but it shouldn't matter
            for (int i = V; i < Vp; i++)
            {
                probs_bt[i] = 0.0f;
            }
        }
    }
}

void attention_forward(float *out, float *preatt, float *att,
                       float *q, float *k, float *v,
                       int B, int T, int C, int NH, int V, int Vp)
{
    // query, key, value (Q, K, V) vectors is (B, T, C(nh*hs))
    // preatt, att are (B, NH, T, T). NH = number of heads, T = sequence length
    // that holds the pre-attention and post-attention scores (used in backward)
    // output is (B, T, C)
    // attention is the only layer that mixes information across time
    // every other operation is applied at every (b,t) position independently
    // (and of course, no layer mixes information across batch)
    int hs = C / NH; // head size
    float scale = 1.0 / sqrtf(hs);
    // NH = 1;

#pragma omp parallel for collapse(3)
    for (int b = 0; b < B; b++)
    {
        for (int t = 0; t < T; t++)
        {
            for (int h = 0; h < NH; h++)
            {
                float *query_t = q + b * T * C + t * C + h * hs;                 // hs
                float *preatt_bth = preatt + b * NH * T * T + h * T * T + t * T; // T, store the plain q.(K_T), not the softmax
                float *att_bth = att + b * NH * T * T + h * T * T + t * T;       // T

                // pass 1: calculate query dot key and maxval
                float maxval = -1000000.0f;     // TODO something better
                for (int t2 = 0; t2 <= t; t2++) // causal self attention
                {
                    float *key_t2 = k + b * T * C + t2 * C + h * hs;

                    // (query_t) dot (key_t2)
                    float val = 0.0f;
                    for (int i = 0; i < hs; i++)
                    {
                        val += query_t[i] * key_t2[i];
                    }
                    val *= scale;
                    if (val > maxval)
                    {
                        maxval = val;
                    }

                    preatt_bth[t2] = val; // just q.K_T
                }
                // softmax_forward(att_bth, preatt_bth, B, T, V, V);

                // pass 2: calculate the exp and keep track of sum
                // maxval is being calculated and subtracted only for numerical stability
                float expsum = 0.0f;
                for (int t2 = 0; t2 <= t; t2++)
                {
                    float expv = expf(preatt_bth[t2] - maxval);
                    expsum += expv;
                    att_bth[t2] = expv;
                }
                float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

                // pass 3: normalize to get the softmax
                for (int t2 = 0; t2 < T; t2++)
                {
                    if (t2 <= t)
                    {
                        att_bth[t2] *= expsum_inv;
                    }
                    else
                    {
                        // causal attention mask. not strictly necessary to set to zero here
                        // only doing this explicitly for debugging and checking to PyTorch
                        att_bth[t2] = 0.0f;
                    }
                }
                // //printf("attn out: %f %f %f %f %f %f %f %f %f %f\n", att_bth[0], att_bth[1], att_bth[2], att_bth[3], att_bth[4], att_bth[5], att_bth[6], att_bth[7], att_bth[8], att_bth[9], att_bth[10]);

                // pass 4: accumulate weighted values into the output of attention
                float *out_bth = out + b * T * C + t * C + h * hs;
                memset(out_bth, 0, hs * sizeof(float));
                for (int i = 0; i < hs; i++)
                {
                    out_bth[i] = 0.0f;
                }
                for (int t2 = 0; t2 <= t; t2++)
                {
                    float *value_t2 = v + b * T * C + t2 * C + h * hs;
                    float att_btht2 = att_bth[t2];
                    for (int i = 0; i < hs; i++)
                    {
                        out_bth[i] += att_btht2 * value_t2[i];
                    }
                }
                // printf("out %d: %f %f %f %f %f %f %f %f %f %f\n", t, out_bth[0], out_bth[1], out_bth[2], out_bth[3], out_bth[4], out_bth[5], out_bth[6], out_bth[7], out_bth[8], out_bth[9], out_bth[10]);
                // //printf("out %d: %f %f %f %f %f %f %f %f %f %f\n", t, out_bth[hs - 1], out_bth[hs - 2], out_bth[hs - 3], out_bth[hs - 4], out_bth[hs - 5], out_bth[hs - 6], out_bth[hs - 7], out_bth[hs - 8], out_bth[hs - 9], out_bth[hs - 10], out_bth[hs - 11]);
            }
        }
    }
    // printf("out: %f %f\n", out[0], out[1]);
    // printf("out: %f %f\n", out[B * T * C - 1], out[B * T * C - 2]);
}

void attention_backward(float *dq, float *dk, float *dv, float *dpreatt, float *datt,
                        float *dout, float *q, float *k, float *v, float *att,
                        int B, int T, int C, int NH)
{
    // inp/dinp are (B, T, 3C) Q,K,V
    // att/datt/dpreatt are (B, NH, T, T)
    // dout is (B, T, C)
    // int C3 = C * 3;
    int hs = C / NH; // head size
    float scale = 1.f / sqrtf(hs);

    for (int b = 0; b < B; b++)
    {
        for (int t = 0; t < T; t++)
        {
            for (int h = 0; h < NH; h++)
            {
                float *att_bth = att + b * NH * T * T + h * T * T + t * T;
                float *datt_bth = datt + b * NH * T * T + h * T * T + t * T;
                float *dpreatt_bth = dpreatt + b * NH * T * T + h * T * T + t * T;
                float *dquery_t = dq + b * T * C + t * C + h * hs;
                float *query_t = q + b * T * C + t * C + h * hs;

                // backward pass 4, through the value accumulation
                float *dout_bth = dout + b * T * C + t * C + h * hs;
                for (int t2 = 0; t2 <= t; t2++)
                {
                    float *value_t2 = v + b * T * C + t2 * C + h * hs;
                    float *dvalue_t2 = dv + b * T * C + t2 * C + h * hs;
                    for (int i = 0; i < hs; i++)
                    {
                        // in the forward pass this was:
                        // out_bth[i] += att_bth[t2] * value_t2[i];
                        // so now we have:
                        datt_bth[t2] += value_t2[i] * dout_bth[i];
                        dvalue_t2[i] += att_bth[t2] * dout_bth[i];
                    }
                }

                // backward pass 2 & 3, the softmax
                // note that softmax (like e.g. tanh) doesn't need the input (preatt) to backward
                for (int t2 = 0; t2 <= t; t2++)
                {
                    for (int t3 = 0; t3 <= t; t3++)
                    {
                        float indicator = t2 == t3 ? 1.0f : 0.0f;
                        float local_derivative = att_bth[t2] * (indicator - att_bth[t3]);
                        // printf("before %d %d %d %d %d l_d= %.8f dattn= %.8f dpre= %.15f\n", b, t, h, t2, t3, local_derivative, datt_bth[t2], dpreatt_bth[t3]);
                        dpreatt_bth[t3] += (local_derivative * datt_bth[t2]);
                        // printf("after %d %d %d %d %d l_d= %.8f dattn= %.8f dpre= %.15f\n", b, t, h, t2, t3, local_derivative, datt_bth[t2], dpreatt_bth[t3]);
                    }
                }

                // backward pass 1, the query @ key matmul
                for (int t2 = 0; t2 <= t; t2++)
                {
                    float *key_t2 = k + b * T * C + t2 * C + h * hs;
                    float *dkey_t2 = dk + b * T * C + t2 * C + h * hs;
                    for (int i = 0; i < hs; i++)
                    {
                        // in the forward pass this was:
                        // preatt_bth[t2] += (query_t[i] * key_t2[i]) * scale;
                        // so now we have:
                        dquery_t[i] += key_t2[i] * dpreatt_bth[t2] * scale;
                        dkey_t2[i] += query_t[i] * dpreatt_bth[t2] * scale;

                        // printf("%d %d %d %d %d k= %.10f dpre= %.15f dk= %.15f\n", b, t, h, t2, i, key_t2[i], dpreatt_bth[t2], dkey_t2[i]);
                        // printf("%d %d %d %d %d q= %.10f dpre= %.10f dq= %.10f\n", b, t, h, t2, i, query_t[i], dpreatt_bth[t2], dquery_t[i]);
                    }
                }
            }
        }
    }
}

void residual_forward(float *out, float *inp1, float *inp2, int N)
{
    for (int i = 0; i < N; i++)
    {
        out[i] = inp1[i] + inp2[i];
    }
}

void residual_backward(float *dinp1, float *dinp2, float *dout, int N)
{
    for (int i = 0; i < N; i++)
    {
        dinp1[i] += dout[i];
        dinp2[i] += dout[i];
    }
}

void crossentropy_forward(float *losses,
                          float *probs, int *targets,
                          int B, int T, int Vp)
{
    // output: losses is (B,T) of the individual losses at each position
    // input: probs are (B,T,Vp) of the probabilities
    // input: targets is (B,T) of integers giving the correct index in logits
    for (int b = 0; b < B; b++)
    {
        for (int t = 0; t < T; t++)
        {
            // loss = -log(probs[target])
            float *probs_bt = probs + b * T * Vp + t * Vp;
            int ix = targets[b * T + t];
            losses[b * T + t] = -logf(probs_bt[ix]);
        }
    }
}

void crossentropy_softmax_backward(float *dlogits,
                                   float *dlosses, float *probs, int *targets,
                                   int B, int T, int V, int Vp)
{
    // backwards through both softmax and crossentropy
    for (int b = 0; b < B; b++)
    {
        for (int t = 0; t < T; t++)
        {
            float *dlogits_bt = dlogits + b * T * Vp + t * Vp;
            float *probs_bt = probs + b * T * Vp + t * Vp;
            float dloss = dlosses[b * T + t];
            int ix = targets[b * T + t];
            // note we only loop to V, leaving the padded dimensions
            // of dlogits untouched, so gradient there stays at zero
            for (int i = 0; i < V; i++)
            {
                float p = probs_bt[i];
                float indicator = i == ix ? 1.0f : 0.0f;
                dlogits_bt[i] += (p - indicator) * dloss;
            }
        }
    }
}

// size is the dim
void matmul(float *xout, float *x, float *w, int n, int d)
{
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    int i;
#pragma omp parallel for private(i)
    for (i = 0; i < d; i++)
    {
        float val = 0.0f;
        for (int j = 0; j < n; j++)
        {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

void rope_forward(float *qr, float *kr, float *q, float *k, int B, int T, int dim, int kv_dim, int n_heads)
{
    // RoPE relative positional encoding: complex-valued rotate q and k in each head
    // qr, q : B,T,C(dim)
    // kr, k: B, T, kv_dim
    int head_size = dim / n_heads;
    for (int b = 0; b < B; b++)
        for (int t = 0; t < T; t++)
            for (int i = 0; i < dim; i += 2)
            {
                int head_dim = i % head_size;
                float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
                float val = t * freq;
                float fcr = cosf(val);
                float fci = sinf(val);
                int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
                for (int v = 0; v < rotn; v++)
                {
                    float *vec = v == 0 ? q : k; // the vector to rotate (query or key)
                    float *rot_vec = v == 0 ? qr : kr;
                    float v0 = vec[b * T * dim + t * dim + i];
                    float v1 = vec[b * T * dim + t * dim + i + 1];
                    rot_vec[b * T * dim + t * dim + i] = v0 * fcr - v1 * fci;
                    rot_vec[b * T * dim + t * dim + i + 1] = v0 * fci + v1 * fcr;
                }
            }
}

void rope_backward(float *dqr, float *dkr, float *dq, float *dk, int B, int T, int dim, int kv_dim, int n_heads)
{
    // RoPE relative positional encoding: complex-valued rotate q and k in each head
    // qr, q : B,T,C(dim)
    // kr, k: B, T, kv_dim
    int head_size = dim / n_heads;
    for (int b = 0; b < B; b++)
        for (int t = 0; t < T; t++)
            for (int i = 0; i < dim; i += 2)
            {
                int head_dim = i % head_size;
                float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
                float val = t * freq;
                float fcr = cosf(val);
                float fci = sinf(val);
                int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
                for (int v = 0; v < rotn; v++)
                {
                    float *dvec = v == 0 ? dq : dk; // the vector to rotate (query or key)
                    float *drot_vec = v == 0 ? dqr : dkr;
                    float *dv0 = dvec + b * T * dim + t * dim + i;
                    float *dv1 = dvec + b * T * dim + t * dim + i + 1;

                    float drv0 = drot_vec[b * T * dim + t * dim + i];
                    float drv1 = drot_vec[b * T * dim + t * dim + i + 1];
                    *dv0 += drv0 * fcr + drv1 * fci;
                    *dv1 += -drv0 * fci + drv1 * fcr;
                }
            }
}

void repeat_kv_forward(float *korv, float *out, int B, int T, int dim, int n_kv_heads, int rep)
{
    // korv : (b,t,n_kv_heads, dim)
    // out : (b,t, n_kv_head * n_rep, dim) -> each kv_head is repeted n_rep times
    // out[b,t,h,d] = korv[b,t,h/rep,d]
    int nh = n_kv_heads * rep;
    int T_kvh_dim = T * n_kv_heads * dim;
    int T_nh_dim = T * nh * dim;

    int nh_dim = nh * dim;
    int N = B * T * nh * dim;

    for (int i = 0; i < N; i++)
    {
        int b = i / T_nh_dim;
        int t = (i % T_nh_dim) / nh_dim;
        int h = (i % nh_dim) / dim;
        int c = i % dim;

        int kv_h = h / rep;
        out[i] = korv[b * T_kvh_dim + t * n_kv_heads * dim + kv_h * dim + c];
    }
}

void repeat_kv_backward(float *dout, float *dx, int B, int T, int n_kv_heads, int rep, int dim)
{
    int nh = n_kv_heads * rep;
    int T_kvh_dim = T * n_kv_heads * dim;
    int T_nh_dim = T * nh * dim;
    int nh_dim = nh * dim;
    int N = B * T * nh * dim;
    for (int i = 0; i < N; i++)
    {
        int b = i / T_nh_dim;
        int t = (i % T_nh_dim) / nh_dim;
        int h = (i % nh_dim) / dim;
        int c = i % dim;
        int kv_h = h / rep;
        dx[b * T_kvh_dim + t * n_kv_heads * dim + kv_h * dim + c] += dout[i];
    }
    return dx;
}
void sum_vector(float *v, int N, float value, int layer, char *name)
{
#ifdef DEBUG
    char *full_name = malloc(strlen(name) + 5);
    sprintf(full_name, "%d ", layer);
    strcat(full_name, name);
    float sum = 0.0;
    for (int i = 0; i < N; i++)
        sum += v[i];

    if (value == 0.0)
    {

        if (fabs(sum) > 1e-8)
        {
            printf("%s_sum: %f\n", full_name, sum);
            assert(fabs(sum) <= 1e-8);
        }
    }
    else
        printf("%s_sum: %f\n", full_name, sum);
    free(full_name);
#endif
}

float *llama2_forward(Transformer *transformer, int *inputs, int *targets, int B, int T)
{
    // B number of sentences, T seq len. must be less than model max seq len
    assert(transformer->config.seq_len >= T);

    int size_per_activation = 1; // act,grad_act when targets != NUll
    malloc_train_state(&transformer->train_state, &transformer->config, B, T);
    Config *p = &(transformer->config);
    int V = p->vocab_size;
    int L = p->n_layers;
    int n_heads = p->n_heads;
    int n_kv_heads = p->n_kv_heads;
    int dim = p->dim;               // C
    int hidden_dim = p->hidden_dim; // This is for the feedforward network

    int head_size = dim / p->n_heads;
    int kv_dim = (head_size * p->n_kv_heads); // this is for the attention heads

    // validate the tokens ids in inputs
    for (int i = 0; i < B * T; i++)
    {
        assert(0 <= inputs[i] && inputs[i] < V);
        if (targets != NULL)
            assert(0 <= targets[i] && targets[i] < V);
    }
    TransformerWeights weights = transformer->weights;
    TransformerWeights weight_grad = transformer->dweights;

    transformer->batch_size = B;
    transformer->seq_len = T;
    transformer->inputs = inputs;
    transformer->targets = targets;

    // encode
    TrainState state = transformer->train_state;
    float *x = state.x;                         // B,T,C
    int *in = transformer->inputs;              // B,T
    float *wte = weights.token_embedding_table; // vocab_size, dim
    int C = dim;
    int OC = p->hidden_dim;
    encoder_forward(x, in, wte, B, T, C); // x -> (B,T,C)

    float *layer_input = x; // input to first layer is the encoding

    for (int l = 0; l < L; l++)
    {
        // rms_attention_norm
        float *weight = weights.rms_att_weight + l * C;
        float *x_rms_attn_norm = state.x_rms_attn_norm + l * B * T * C; // C
        rmsnorm_forward(x_rms_attn_norm, layer_input, weight, B, T, C); // xb2 -> (B,T,C)

        float *q = state.q + l * B * T * C;      // q (B,T,dim,)
        float *k = state.k + l * B * T * kv_dim; // k (B,T,kv_dim,)
        float *v = state.v + l * B * T * kv_dim; // v (B,T,kv_dim,)
                                                 // rope

        // attention forward(rms_attention_norm)
        matmul_forward_naive(q, x_rms_attn_norm, weights.wq + l * dim * dim, NULL, B, T, C, C);
        matmul_forward_naive(k, x_rms_attn_norm, weights.wk + l * dim * kv_dim, NULL, B, T, C, kv_dim);
        matmul_forward_naive(v, x_rms_attn_norm, weights.wv + l * dim * kv_dim, NULL, B, T, C, kv_dim);

        float *qr = state.qr + l * B * T * dim;    // qr (B,T,dim,)
        float *kr = state.kr + l * B * T * kv_dim; // kr (B,T,dim,)
        sum_vector(kr, B * T * kv_dim, 0.0, l, "kr");
        rope_forward(qr, kr, q, k, B, T, dim, kv_dim, n_heads);

        // causal attention
        float *out = state.attn_out + l * B * T * C;                              // B,T,C
        float *xo = state.xo + l * B * T * C;                                     // B,T,C
        float *x_res = state.x_res + l * B * T * C;                               // B,T,C
        float *attn = state.att + l * B * n_heads * T * T;                        // B,NH,T,T
        float *preattn = state.preatt + l * B * n_heads * T * T;                  // B,NH,T,T
        attention_forward(out, preattn, attn, qr, kr, v, B, T, C, n_heads, V, V); // out : B,T,C
        float *wo = weights.wo + l * dim * dim;                                   // wo : dim, dim
        matmul_forward_naive(xo, out, wo, NULL, B, T, C, C);
        residual_forward(x_res, xo, layer_input, B * T * C);
        weight = weights.rms_ffn_weight + l * C;
        float *x_res_rms_ffn_norm = state.x_res_rms_ffn_norm + l * B * T * C;
        rmsnorm_forward(x_res_rms_ffn_norm, x_res, weight, B, T, C);

        float *w1 = weights.w1 + l * dim * hidden_dim;
        float *w2 = weights.w2 + l * dim * hidden_dim;
        float *w3 = weights.w3 + l * dim * hidden_dim;
        float *h1 = state.h1 + l * B * T * OC;
        float *h1_silu = state.h1_silu + l * B * T * OC;

        float *h3 = state.h3 + l * B * T * OC;
        float *h1_h3_prod = state.h1_h3_prod + l * B * T * OC;
        float *h2 = state.h2 + l * B * T * C;
        float *x_ffn_out = state.x_ffn + l * B * T * C;
        feedforward(x_ffn_out, h1, h2, h3, h1_silu, h1_h3_prod, x_res_rms_ffn_norm, w1, w2, w3, B, T, C, OC);

        float *x_final_res = state.x_final_res + l * B * T * C;
        residual_forward(x_final_res, x_res, x_ffn_out, B * T * C);
        layer_input = x_final_res;
    }

    float *weight = weights.rms_final_weight;
    float *x_final_res = state.x_final_res + (L - 1) * B * T * C;
    rmsnorm_forward(state.x_final_rms_norm, x_final_res, weight, B, T, C);

    matmul_forward_naive(state.logits, state.x_final_rms_norm, weights.wcls, NULL, B, T, C, p->vocab_size);
    softmax_forward(state.prob, state.logits, B, T, V, V);

    // also forward the cross-entropy loss function if we have the targets
    float mean_loss = 0.0f;
    if (targets != NULL)
    {
        crossentropy_forward(state.loss, state.prob, targets, B, T, V);
        // for convenience also evaluate the mean loss

        for (int i = 0; i < B * T; i++)
        {
            mean_loss += state.loss[i];
        }
        mean_loss /= B * T;
    }
    else
    {
        // if we don't have targets, we don't have a loss
        mean_loss = -1.0f;
    }
    printf("mean loss: %f\n", mean_loss);
    return state.logits;
}

void llama2_backward(Transformer *model)
{
    // loss backward into grad_loss Transformer->train_state.loss[i]
    // crossentropy_softmax_backward() : grad_loss -> grad_logits
    // matmul_backward : grad_logits -> grad_x_final_rms_norm, grad_wcls
    // rms_backward (1115): grad_x_final_rms_norm -> grad_x_final_res, grad_rms_final_weight

    // backprop thru the layers (L-1 to 0)
    //      residual_backward(1096): grad_x_final_res -> grad_x_res, grad_x_ffn_out
    //      feedforward_backward(1092): grad_x_ffn_out -> grad_h1, grad_h2, grad_h3, grad_h1_silu, grad_h1_h3_prod, grad_x_res_rms_ffn_norm, grad_w1, grad_w2, grad_w3
    //      rmsnorm_forward(1079): grad_x_res_rms_ffn_norm -> grad_x_res, grad_rms_ffn_weight
    //      residaul_backward(1075) : grad_x_res -> grad_xo, grad_layer_input
    //      matmul_backward(1072): grad_xo -> grad_out, grad_wo
    //      attention_forward(1069): grad_out -> grad_preattn, grad_attn, grad_qr, grad_kr, grad_v
    //      rope_forward(1059): grad_qr, grad_kr -> grad_q, grad_k
    //      matmul_backward(1045-1047): grad_q -> grad_x_rms_attn_norm, grad_wq
    //                                  grad_k -> grad_x_rms_attn_norm, grad_wk
    //                                  grad_v -> grad_x_rms_attn_norm, grad_wv
    //      rmsnorm_forward(1035): grad_x_rms_attn_norm -> grad_layer_input, grad_rms_attn_weight

    //  encoder_forward -> grad_layer_0_input -> grad_in, grad_wte

    int B = model->batch_size;
    int T = model->seq_len;
    int C = model->config.dim;
    int hidden_dim = model->config.hidden_dim;
    int NH = model->config.n_heads;
    int n_kv_heads = model->config.n_kv_heads;
    int L = model->config.n_layers;
    int V = model->config.vocab_size;

    // allocate grad activations
    TrainState *grad = &model->grad_train_state;
    TrainState *state = &model->train_state;
    malloc_train_state(grad, &model->config, B, T);
    // exit(1);
    TransformerWeights *dweights = &model->dweights;
    TransformerWeights *weights = &model->weights;

    int grad_offset = B * T;
    float dloss_mean = 1.0f / (B * T);
    float *dlosses = grad->loss; // B,T
    for (int i = 0; i < B * T; i++)
        dlosses[i] = dloss_mean;

    // cross entropy softmax
    float *probs = state->prob;
    float *dlogits = grad->logits;
    crossentropy_softmax_backward(dlogits, dlosses, probs, model->targets, B, T, V, V);

    float *dlayer_input; // = (float *)calloc(B * T * C, sizeof(float));
    // matmul_forward_naive(state.logits, state.x_final_rms_norm, weights.wcls, NULL, B, T, C, p->vocab_size);
    float *inp = state->x_final_rms_norm;
    float *dinp = grad->x_final_rms_norm;
    float *wcls = weights->wcls;
    float *dwcls = dweights->wcls;
    matmul_backward(dinp, dwcls, NULL, dlogits, inp, wcls, B, T, C, V);
    printf("dwcls: %.11f %.11f %.11f\n", dwcls[0], dwcls[1], dwcls[C * V - 1]);

    // rmsnorm backward
    // rmsnorm_forward(state.x_final_rms_norm, x_final_res, weight, B, T, C);
    float *x = state->x_final_res + (L - 1) * B * T * C;
    float *dx_final_res = grad->x_final_res + (L - 1) * B * T * C;
    sum_vector(dx_final_res, B * T * C, 0.0, L - 1, "grad->x_final_res");
    dlayer_input = dx_final_res;
    float *dwrms = dweights->rms_final_weight;
    float *w = weights->rms_final_weight;
    rmsnorm_backward(dinp, dx_final_res, dwrms, x, w, B, T, C);
    printf("dwrms: %f %f\n", dwrms[0], dwrms[C - 1]);

    // backprop thru the layers
    for (int l = L - 1; l >= 0; l--)
    {
        // residual backward
        // float *x_final_res = state.x_final_res + l *  B * T * C;
        // residual_forward(x_final_res, x_res, x_ffn_out, B * T * C);
        // float *dout = dx_final_res;
        sum_vector(grad->x_rms_attn_norm + l * B * T * C, B * T * C, 0.0, l, "grad->x_rms_attn_norm");
        float *dout = dlayer_input;
        float *dinp_skip = grad->x_res + l * (B * T * C);

        sum_vector(dinp_skip, B * T * C, 0.0, l, "grad->x_res");
        sum_vector(grad->x_rms_attn_norm + l * B * T * C, B * T * C, 0.0, l, "grad->x_rms_attn_norm");
        float *dinp = grad->x_ffn + l * (B * T * C);
        sum_vector(dinp, B * T * C, 0.0, l, "grad->x_ffn");
        sum_vector(grad->x_rms_attn_norm + l * B * T * C, B * T * C, 0.0, l, "grad->x_rms_attn_norm");
        residual_backward(dinp, dinp_skip, dout, B * T * C);

        // ffn backward
        float *w1 = weights->w1 + l * C * hidden_dim;
        float *dw1 = dweights->w1 + l * C * hidden_dim;
        float *w2 = weights->w2 + l * C * hidden_dim;
        float *dw2 = dweights->w2 + l * C * hidden_dim;
        float *w3 = weights->w3 + l * C * hidden_dim;
        float *dw3 = dweights->w3 + l * C * hidden_dim;
        float *h1 = state->h1 + l * B * T * hidden_dim;
        float *dh1 = grad->h1 + l * B * T * hidden_dim;
        sum_vector(dh1, B * T * hidden_dim, 0.0, l, "dh1");
        sum_vector(grad->x_rms_attn_norm + l * B * T * C, B * T * C, 0.0, l, "grad->x_rms_attn_norm");
        float *h1_silu = state->h1_silu + l * B * T * hidden_dim;
        float *dh1_silu = grad->h1_silu + l * B * T * hidden_dim;
        sum_vector(dh1_silu, B * T * hidden_dim, 0.0, l, "dh1_sliu");
        sum_vector(grad->x_rms_attn_norm + l * B * T * C, B * T * C, 0.0, l, "grad->x_rms_attn_norm");
        float *h3 = state->h3 + l * B * T * hidden_dim;
        float *dh3 = grad->h3 + l * B * T * hidden_dim;
        sum_vector(dh3, B * T * hidden_dim, 0.0, l, "dh3");
        sum_vector(grad->x_rms_attn_norm + l * B * T * C, B * T * C, 0.0, l, "grad->x_rms_attn_norm");

        float *h1_h3_prod = state->h1_h3_prod + l * B * T * hidden_dim;
        float *dh1_h3_prod = grad->h1_h3_prod + l * B * T * hidden_dim;
        sum_vector(dh1_h3_prod, B * T * hidden_dim, 0.0, l, "dh1_h3_prod");
        sum_vector(grad->x_rms_attn_norm + l * B * T * C, B * T * C, 0.0, l, "grad->x_rms_attn_norm");
        float *h2 = state->h2 + l * B * T * C;
        float *dh2 = grad->h2 + l * B * T * C;
        sum_vector(dh2, B * T * C, 0.0, l, "dh2");
        sum_vector(grad->x_rms_attn_norm + l * B * T * C, B * T * C, 0.0, l, "grad->x_rms_attn_norm");
        float *x_ffn_out = state->x_ffn + l * B * T * C;

        // feedforward(x_ffn_out, h1, h2, h3, h1_silu, h1_h3_prod, x_res_rms_ffn_norm, w1, w2, w3, B, T, C, OC);
        dout = dinp;
        dinp = grad->x_res_rms_ffn_norm + l * B * T * C;
        sum_vector(dinp, B * T * C, 0.0, l, "grad->x_res_rms_ffn_norm");
        sum_vector(grad->x_rms_attn_norm + l * B * T * C, B * T * C, 0.0, l, "grad->x_rms_attn_norm");
        float *inp = state->x_res_rms_ffn_norm + l * B * T * C;
        feedforward_backward(dout, dh1, dh2, dh1_silu, dh1_h3_prod, dh3, h1, h3, h1_silu, h1_h3_prod, dinp, dw1, dw2, dw3, inp, x_ffn_out, w1, w2, w3, B, T, C, hidden_dim);
        printf("l: %d, dw1: %f %f\n", l, dw1[0], dw1[C * hidden_dim - 1]);
        printf("l: %d, dw2: %f %f\n", l, dw2[0], dw2[C * hidden_dim - 1]);
        printf("l: %d, dw3: %f %f\n", l, dw3[0], dw3[C * hidden_dim - 1]);
        if (l > 0)
            sum_vector(grad->x_rms_attn_norm + l * B * T * C - B * T * C, B * T * C, 0.0, l, "[prev] grad->x_rms_attn_norm");
        // rms norm backward
        // weight = weights.rms_ffn_weight + l * C;
        // float *x_res_rms_ffn_norm = state.x_res_rms_ffn_norm + l *  B * T * C;
        // rmsnorm_forward(x_res_rms_ffn_norm, x_res, weight, B, T, C);
        // dout = dinp;
        x = state->x_res + l * B * T * C;
        dout = dinp;
        dinp = grad->x_res + l * (B * T * C);
        sum_vector(dinp, B * T * C, -1.0, l, "again grad->x_res");
        sum_vector(grad->x_rms_attn_norm + l * B * T * C, B * T * C, 0.0, l, "grad->x_rms_attn_norm");
        if (l > 0)
            sum_vector(grad->x_rms_attn_norm + l * B * T * C - B * T * C, B * T * C, 0.0, l, "[prev] grad->x_rms_attn_norm");
        float *dw_rms_ffn_weight = dweights->rms_ffn_weight + l * C;
        float *rms_ffn_weight = weights->rms_ffn_weight + l * C;
        rmsnorm_backward(dout, dinp, dw_rms_ffn_weight, x, rms_ffn_weight, B, T, C);
        printf("l: %d, dw_rms_ffn_weight: %f %f\n", l, dw_rms_ffn_weight[0], dw_rms_ffn_weight[C - 1]);

        // residual backward
        // residual_forward(x_res, xo, layer_input, B * T * C);
        dout = dinp;
        dinp = grad->xo + l * B * T * C;
        sum_vector(dinp, B * T * C, 0.0, l, "grad->xo");
        sum_vector(grad->x_rms_attn_norm + l * B * T * C, B * T * C, 0.0, l, "grad->x_rms_attn_norm");
        dinp_skip = l > 0 ? grad->x_final_res + (l - 1) * B * T * C : grad->x;
        sum_vector(dinp_skip, B * T * C, 0.0, l, "grad->x");

        sum_vector(grad->x_rms_attn_norm + l * B * T * C, B * T * C, 0.0, l, "grad->x_rms_attn_norm");
        if (l > 0)
            sum_vector(grad->x_rms_attn_norm + l * B * T * C - B * T * C, B * T * C, 0.0, l, "[prev] grad->x_rms_attn_norm");
        residual_backward(dinp, dinp_skip, dout, B * T * C);
        if (l > 0)
            sum_vector(grad->x_rms_attn_norm + l * B * T * C - B * T * C, B * T * C, 0.0, l, "[prev] grad->x_rms_attn_norm");
        // matmul_backward
        float *wo = weights->wo + l * C * C;
        float *dwo = dweights->wo + l * C * C; // wo : dim, dim
        dout = dinp;
        inp = state->attn_out + l * B * T * C;
        dinp = grad->attn_out + l * B * T * C;
        sum_vector(dinp, B * T * C, 0.0, l, "grad->attn_out");
        sum_vector(grad->x_rms_attn_norm + l * B * T * C, B * T * C, 0.0, l, "grad->x_rms_attn_norm");
        // matmul_forward_naive(xo, out, wo, NULL, B, T, C, C);
        matmul_backward(dinp, dwo, NULL, dout, inp, wo, B, T, C, C);
        printf("l: %d, dwo: %f %f\n", l, dwo[0], dwo[C * C - 1]);
        if (l > 0)
            sum_vector(grad->x_rms_attn_norm + l * B * T * C - B * T * C, B * T * C, 0.0, l, "[prev] grad->x_rms_attn_norm");
        // backprop attention
        int head_size = C / NH;
        int kv_dim = n_kv_heads * head_size;

        printf("nh: %d, n_kv_heads: %d, head_size: %d, dim: %d, kv_dim: %d\n", NH, n_kv_heads, head_size, C, kv_dim);
        float *qr = state->qr + l * B * T * C;      // qr (B,T,dim,)
        float *kr = state->kr + l * B * T * kv_dim; // kr (B,T,dim,)
        float *v = state->v + l * B * T * kv_dim;   // qr (B,T,dim,)
        float *dqr = grad->qr + l * B * T * C;      // qr (B,T,dim,)
        float *dkr = grad->kr + l * B * T * kv_dim; // kr (B,T,dim,)
        float *dv = grad->v + l * B * T * kv_dim;   // qr (B,T,dim,)

        float *out = state->attn_out + l * B * T * C;        // B,T,C
        float *attn = state->att + l * B * NH * T * T;       // B,NH,T,T
        float *preattn = state->preatt + l * B * NH * T * T; // B,NH,T,T
        dout = dinp;
        float *datt = grad->att + l * B * NH * T * T;       // B,NH,T,T
        float *dpreatt = grad->preatt + l * B * NH * T * T; // B,NH,T,T

        sum_vector(dqr, B * T * C, 0.0, l, "grad->qr");
        sum_vector(dkr, B * T * kv_dim, 0.0, l, "grad->kr");
        sum_vector(dv, B * T * kv_dim, 0.0, l, "grad->v");
        sum_vector(datt, B * NH * T * T, 0.0, l, "grad->attn");
        sum_vector(dpreatt, B * NH * T * T, 0.0, l, "grad->preatt");
        sum_vector(grad->x_rms_attn_norm + l * B * T * C, B * T * C, 0.0, l, "grad->x_rms_attn_norm");
        if (l > 0)
            sum_vector(grad->x_rms_attn_norm + l * B * T * C - B * T * C, B * T * C, 0.0, l, "[prev] grad->x_rms_attn_norm");
        attention_backward(dqr, dkr, dv, dpreatt, datt,
                           dout, qr, kr, v, attn,
                           B, T, C, NH);
        printf("l: %d, q: %f %f\n", l, qr[0], qr[B * T * C - 1]);
        printf("l: %d, k: %f %f\n", l, kr[0], kr[B * T * kv_dim - 1]);
        printf("l: %d, dout: %f %f\n", l, dout[0], dout[B * T * C - 1]);
        printf("l: %d, datt: %f %f\n", l, datt[0], datt[B * NH * T * T - 1]);
        printf("l: %d, dpreatt: %f %f\n", l, dpreatt[0], dpreatt[B * NH * T * T - 1]);
        printf("l: %d, dout: %f %f\n", l, dout[0], dout[B * T * C - 1]);
        printf("l: %d, dq: %f %f %f\n", l, dqr[0], dqr[1], dqr[B * T * C - 1]);
        printf("l: %d, dk: %f %f %f\n", l, dkr[0], dkr[1], dkr[B * T * kv_dim - 1]);
        printf("l: %d, dv: %f %f %f\n", l, dv[0], dv[1], dv[B * T * kv_dim - 1]);
        // matmul wq,wk,wv backward

        // rope backward
        float *q = state->q + l * B * T * C;      // qr (B,T,dim,)
        float *k = state->k + l * B * T * kv_dim; // kr (B,T,dim,)
        float *dq = grad->q + l * B * T * C;      // qr (B,T,dim,)
        float *dk = grad->k + l * B * T * kv_dim; // kr (B,T,dim,)
        sum_vector(dq, B * T * C, 0.0, l, "dq");
        sum_vector(dk, B * T * kv_dim, 0.0, l, "dk");
        sum_vector(grad->x_rms_attn_norm + l * B * T * C, B * T * C, 0.0, l, "dx_rms_attn_norm");
        if (l > 0)
            sum_vector(grad->x_rms_attn_norm + l * B * T * C - B * T * C, B * T * C, 0.0, l, "[prev] grad->x_rms_attn_norm");
        rope_backward(dqr, dkr, dq, dk, B, T, C, kv_dim, NH);
        sum_vector(dq, B * T * C, -1.0, l, "dq");
        sum_vector(dk, B * T * kv_dim, -1.0, l, "dk");
        float *dwq = dweights->wq + l * C * C;
        float *dwk = dweights->wk + l * C * kv_dim;
        float *dwv = dweights->wv + l * C * kv_dim;

        // dq = (float*) calloc(B*T*C, sizeof(float));
        // memset((float*)dv, 1, B*T*C);
        if (l > 0)
            sum_vector(grad->x_rms_attn_norm + l * B * T * C - B * T * C, B * T * C, 0.0, l, "[prev] grad->x_rms_attn_norm");
        dinp = grad->x_rms_attn_norm + l * B * T * C;
        if (l > 0)
            sum_vector(grad->x_rms_attn_norm + l * B * T * C - B * T * C, B * T * C, 0.0, l, "[prev] grad->x_rms_attn_norm");
        inp = state->x_rms_attn_norm + l * B * T * C;

        float *wq = weights->wq + l * C * C;
        float *wk = weights->wk + l * C * kv_dim;
        float *wv = weights->wv + l * C * kv_dim;

        float sum_dinp = 0.0;
        sum_vector(dinp, B * T * C, 0.0, l, "before grad->x_rms_attn_norm");
        if (l > 0)
            sum_vector(dinp - B * T * C, B * T * C, 0.0, l, "[prev] grad->x_rms_attn_norm");

        sum_vector(inp, B * T * C, -1.0, l, "x_rms_attn_norm");
        matmul_backward(dinp, dwv, NULL, dv, inp, wv, B, T, C, kv_dim);
        matmul_backward(dinp, dwk, NULL, dk, inp, wk, B, T, C, kv_dim);
        matmul_backward(dinp, dwq, NULL, dq, inp, wq, B, T, C, C);

        if (l > 0)
            sum_vector(grad->x_rms_attn_norm + l * B * T * C - B * T * C, B * T * C, 0.0, l, "[prev] grad->x_rms_attn_norm");
        sum_vector(dinp, B * T * C, -1, l, "after grad->x_rms_attn_norm");

        float sum_wq = 0.0;
        float sum_wv = 0.0;
        float sum_wk = 0.0;

        float sum_dq = 0.0;
        float sum_dv = 0.0;
        float sum_dk = 0.0;

        float sum_inp = 0.0;
        sum_dinp = 0.0;
        for (int j = 0; j < B * T * C; j++)
        {
            sum_inp += inp[j];
            sum_dinp += dinp[j];
            sum_dk += dk[j];
            sum_dq += dq[j];
            sum_dv += dv[j];
        }
        for (int j = 0; j < C * C; j++)
        {
            sum_wq += wq[j];
            sum_wv += wv[j];
            sum_wk += wk[j];
        }

        printf("l: %d, x: %f %f\n", l, x[0], x[B * T * C - 1]);
        printf("l: %d, sum_inp: %f, sum_dinp: %f\n", l, sum_inp, sum_dinp);
        printf("l: %d, wq: %f %f sum_wq: %f, sum_dq: %f\n", l, wq[0], wq[C * C - 1], sum_wq, sum_dq);
        printf("l: %d, wk: %f %f sum_wk: %f, sum_dk: %f\n\n", l, wk[0], wk[C * kv_dim - 1], sum_wk, sum_dk);
        printf("l: %d, wv: %f %f sum_wv: %f, sum_dv: %f\n", l, wv[0], wv[C * kv_dim - 1], sum_wv, sum_dv);
        sum_vector(dwq, C * C, -1, l, "dwq:");
        printf("l: %d, dwq: %f %f\n", l, dwq[0], dwq[C * C - 1]);
        sum_vector(dwk, C * kv_dim, -1, l, "dwk:");
        printf("l: %d, dwk: %f %f\n", l, dwk[0], dwk[C * kv_dim - 1]);
        sum_vector(dwv, C * kv_dim, -1, l, "dwv:");
        printf("l: %d, dwv: %f %f\n", l, dwv[0], dwv[C * kv_dim - 1]);
        sum_vector(dwo, C * C, -1, l, "dwo:");
        printf("l: %d, dinp: %f %f %f\n", l, dinp[0], dinp[1], dinp[B * T * C - 1]);
        // rms norm backward
        float *rms_att_weight = weights->rms_att_weight + l * C;
        float *drms_att_weight = dweights->rms_att_weight + l * C;
        float *x_rms_attn_norm = state->x_rms_attn_norm + l * B * T * C; // C
        x = l > 0 ? state->x_final_res + (l - 1) * B * T * C : state->x;

        rmsnorm_backward(dinp, dinp_skip, drms_att_weight, x, rms_att_weight, B, T, C);
        sum_vector(dinp, B * T * C, -1.0, l, "final grad->x_rms_attn_norm");
        if (l > 0)
            sum_vector(dinp - B * T * C, B * T * C, 0.0, l - 1, "final grad->x_rms_attn_norm");
        printf("l: %d, drms_att_weight: %f %f\n", l, drms_att_weight[0], drms_att_weight[C - 1]);

        /// set the new layer grad
        dlayer_input = dinp_skip;
        printf("l: %d, dlayer_input: %f %f\n", l, dlayer_input[0], dlayer_input[B * T * C - 1]);
    }

    // embeddingding backward
    float *wte = weights->token_embedding_table; // vocab_size, dim
    float *dwte = dweights->token_embedding_table;
    float sum_wte = 0.0;
    float sum_dwte = 0.0;

    // encoder_forward(x, in, wte, B, T, C); // x -> (B,T,C)
    // for (int j = 0; j < C * model->config.vocab_size; j++)
    //     dwte[j] = 0.0;
    encoder_backward(dwte, dlayer_input, model->inputs, B, T, C);
    for (int j = 0; j < C * model->config.vocab_size; j++)
    {
        sum_wte += wte[j];
        sum_dwte += dwte[j];
        // dwte[j] = 0.0;
    }
    printf("wte: %.8f %.8f %.8f\n", wte[0], wte[1], wte[C * model->config.vocab_size - 1]);
    printf("dwte: %.8f %.8f, sum_wte: %f, sum_dwte: %f\n", dwte[0], dwte[C * model->config.vocab_size - 1], sum_wte, sum_dwte);
}
// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

void save_model_state(Transformer *model, int *tokens, int *targets)
{
    // save the grad activations and grad weights for testing
    TrainState *act = &model->train_state;
    TrainState *grad = &model->grad_train_state;

    TransformerWeights *weights = &model->weights;
    TransformerWeights *dweights = &model->dweights;
    Config config = model->config;
    int B = model->batch_size;
    int T = model->seq_len;
    int C = model->config.dim;
    int OC = model->config.hidden_dim;
    int L = model->config.n_layers;
    int nh = model->config.n_heads;
    int V = model->config.vocab_size;
    int n_kv_heads = model->config.n_kv_heads;
    // printf(n_kv_heads);
    // exit(0);

    FILE *fp = fopen("state.bin", "wb");
    if (fp == NULL)
    {
        printf("Error opening file\n");
        exit(1);
    }

    // write the configs
    fwrite(&B, sizeof(int), 1, fp);
    fwrite(&T, sizeof(int), 1, fp);
    fwrite(&C, sizeof(int), 1, fp);
    fwrite(&OC, sizeof(int), 1, fp);
    fwrite(&L, sizeof(int), 1, fp);
    fwrite(&nh, sizeof(int), 1, fp);
    fwrite(&V, sizeof(int), 1, fp);
    fwrite(&n_kv_heads, sizeof(int), 1, fp);

    // write the tokens and targets
    fwrite(tokens, sizeof(int), B * T, fp);
    fwrite(targets, sizeof(int), B * T, fp);

    // write the grad activations
    fwrite(grad->x, sizeof(float), B * T * C, fp);
    // fwrite(grad->layer_input, sizeof(float), B * T * C, fp);
    fwrite(grad->x_rms_attn_norm, sizeof(float), L * B * T * C, fp);

    fwrite(grad->q, sizeof(float), L * B * T * C, fp);
    fwrite(grad->k, sizeof(float), L * B * T * C, fp);
    fwrite(grad->v, sizeof(float), L * B * T * C, fp);

    fwrite(grad->qr, sizeof(float), L * B * T * C, fp);
    fwrite(grad->kr, sizeof(float), L * B * T * C, fp);

    fwrite(grad->att, sizeof(float), L * B * T * T * nh, fp);
    fwrite(grad->preatt, sizeof(float), L * B * T * T * nh, fp);
    fwrite(grad->attn_out, sizeof(float), L * B * T * C, fp);
    fwrite(grad->x_attn, sizeof(float), L * B * T * C, fp);
    fwrite(grad->xo, sizeof(float), L * B * T * C, fp);
    fwrite(grad->x_res, sizeof(float), L * B * T * C, fp);
    fwrite(grad->x_res_rms_ffn_norm, sizeof(float), L * B * T * C, fp);

    fwrite(grad->h1, sizeof(float), L * B * T * OC, fp);
    fwrite(grad->h2, sizeof(float), L * B * T * C, fp);
    fwrite(grad->h3, sizeof(float), L * B * T * OC, fp);
    fwrite(grad->h1_h3_prod, sizeof(float), L * B * T * OC, fp);
    fwrite(grad->h1_silu, sizeof(float), L * B * T * OC, fp);

    fwrite(grad->x_ffn, sizeof(float), L * B * T * C, fp);
    fwrite(grad->x_final_res, sizeof(float), L * B * T * C, fp);
    fwrite(grad->x_final_rms_norm, sizeof(float), B * T * C, fp);

    fwrite(grad->logits, sizeof(float), B * T * V, fp);
    fwrite(grad->prob, sizeof(float), B * T * V, fp);
    fwrite(grad->loss, sizeof(float), B * T, fp);

    // write the grad of weights
    fwrite(dweights->token_embedding_table, sizeof(float), V * C, fp);

    fwrite(dweights->rms_att_weight, sizeof(float), L * C, fp);
    fwrite(dweights->rms_ffn_weight, sizeof(float), L * C, fp);

    int head_size = C / nh;
    fwrite(dweights->wq, sizeof(float), L * C * C, fp);
    fwrite(dweights->wk, sizeof(float), L * C * n_kv_heads * head_size, fp);
    fwrite(dweights->wv, sizeof(float), L * C * n_kv_heads * head_size, fp);
    fwrite(dweights->wo, sizeof(float), L * C * C, fp);

    fwrite(dweights->w1, sizeof(float), L * OC * C, fp);
    fwrite(dweights->w2, sizeof(float), L * OC * C, fp);
    fwrite(dweights->w3, sizeof(float), L * OC * C, fp);

    fwrite(dweights->rms_final_weight, sizeof(float), C, fp);

    // (optional) classifier weights for the logits, on the last layer
    fwrite(dweights->wcls, sizeof(float), C * V, fp);

    fclose(fp);
    printf("Saved grads to state.bin\n");
}
typedef struct
{
    char *str;
    int id;
} TokenIndex;

typedef struct
{
    char **vocab;
    float *vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512]; // stores all single-byte strings
} Tokenizer;

int compare_tokens(const void *a, const void *b)
{
    return strcmp(((TokenIndex *)a)->str, ((TokenIndex *)b)->str);
}

void build_tokenizer(Tokenizer *t, char *tokenizer_path, int vocab_size)
{
    // i should have written the vocab_size into the tokenizer file... sigh
    t->vocab_size = vocab_size;
    // malloc space to hold the scores and the strings
    t->vocab = (char **)malloc(vocab_size * sizeof(char *));
    t->vocab_scores = (float *)malloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL; // initialized lazily
    for (int i = 0; i < 256; i++)
    {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }
    // read in the file
    FILE *file = fopen(tokenizer_path, "rb");
    if (!file)
    {
        fprintf(stderr, "couldn't load %s\n", tokenizer_path);
        exit(EXIT_FAILURE);
    }
    if (fread(&t->max_token_length, sizeof(int), 1, file) != 1)
    {
        fprintf(stderr, "failed read\n");
        exit(EXIT_FAILURE);
    }
    int len;
    for (int i = 0; i < vocab_size; i++)
    {
        if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1)
        {
            fprintf(stderr, "failed read\n");
            exit(EXIT_FAILURE);
        }
        if (fread(&len, sizeof(int), 1, file) != 1)
        {
            fprintf(stderr, "failed read\n");
            exit(EXIT_FAILURE);
        }
        t->vocab[i] = (char *)malloc(len + 1);
        if (fread(t->vocab[i], len, 1, file) != 1)
        {
            fprintf(stderr, "failed read\n");
            exit(EXIT_FAILURE);
        }
        t->vocab[i][len] = '\0'; // add the string terminating token
    }
    fclose(file);
}

void free_tokenizer(Tokenizer *t)
{
    for (int i = 0; i < t->vocab_size; i++)
    {
        free(t->vocab[i]);
    }
    free(t->vocab);
    free(t->vocab_scores);
    free(t->sorted_vocab);
}

int main(int argc, char *argv[])
{
    // B,T,n_kv_heads, dim,rep = 2,3,2,5,4
    if (argc > 1 && strcmp(argv[1], "repeat") == 0)
    {
#define B 4
#define T 30
#define n_kv_heads 5
#define C 50
#define rep 4
        int dim = (int)C;
        float *k = (float *)calloc(B * T * n_kv_heads * dim, sizeof(float));
        float *out = (float *)calloc(B * T * n_kv_heads * rep * dim, sizeof(float));
        printf("%d %d %d %d %d", B, T, n_kv_heads, dim, rep);
        printf("****");
        for (int i = 0; i < B * T * n_kv_heads * dim; i++)
        {
            k[i] = rand() / (float)RAND_MAX;
            printf("%f ", k[i]);
        }
        printf("****");
        repeat_kv_forward(k, out, B, T, dim, n_kv_heads, rep);
        for (int i = 0; i < B * T * n_kv_heads * rep * dim; i++)
            printf("%f ", out[i]);

        printf("****");
        float *dk = (float *)calloc(B * T * n_kv_heads * dim, sizeof(float));
        repeat_kv_backward(out, dk, B, T, n_kv_heads, rep, dim);
        for (int i = 0; i < B * T * n_kv_heads * dim; i++)
            printf("%f ", dk[i]);
        printf("****");
        return 0;
    }
    Transformer transformer;
    // char *checkpoint_path = "gpt2_124M.bin"; //"stories15M.bin";
    char *checkpoint_path = "llama_test.bin";
    build_transformer(&transformer, checkpoint_path);
    printf("Loaded new llama model\n");

    int tokens[2][5] = {
        {1, 3439, 17632, 1925, 29871},
        {1, 2, 7632, 125, 29871}};
    int targets[2][5] = {
        {1, 3439, 17632, 1925, 29871},
        {1, 2, 7632, 125, 29871}};
    const int seq_len = 5;
    float lr = 0.001;
    float beta1 = 0.9;
    float beta2 = 0.999;
    float eps = 1e-08;
    float weight_decay = 0.01;
    Config c = transformer.config;

#define C c.dim
#define V c.vocab_size
#define L c.n_layers
#define NH c.n_heads
#define n_kv_heads c.n_kv_heads
#define OC c.hidden_dim

    llama2_forward(&transformer, (int *)tokens, (int *)targets, 2, seq_len);
    printf("Done forward pass\n");
    llama2_backward(&transformer);
    printf("Done backward pass\n");
    // update_weights(&transformer, lr, beta1, beta2, eps, weight_decay, 1);

    int head_size = (int)C / (int)NH;
    printf("V: %d, C: %d\n", V, C);
    sum_vector(transformer.weights.token_embedding_table, V * C, -1.0, -1, "wte");
    sum_vector(transformer.weights.rms_att_weight, L * C, -1.0, -1, "rms_att_weight");
    sum_vector(transformer.weights.rms_ffn_weight, L * C, -1.0, -1, "rms_ffn_weight");
    sum_vector(transformer.weights.wq, L * C * C, -1.0, -1, "wq");
    sum_vector(transformer.weights.wk, L * C * n_kv_heads * head_size, -1.0, -1, "wk");
    sum_vector(transformer.weights.wv, L * C * n_kv_heads * head_size, -1.0, -1, "wv");
    sum_vector(transformer.weights.wo, L * C * C, -1.0, -1, "wo");
    sum_vector(transformer.weights.w1, L * C * OC, -1.0, -1, "w1");
    sum_vector(transformer.weights.w2, L * C * OC, -1.0, -1, "w2");
    sum_vector(transformer.weights.w3, L * C * OC, -1.0, -1, "w3");
    sum_vector(transformer.weights.rms_final_weight, C, -1.0, -1, "rms_final_weight");
    save_model_state(&transformer, (int *)tokens, (int *)targets);
    free_trained_transformer(&transformer);
}
