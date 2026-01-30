#include "llaisys/models/qwen2.h"

#include "../llaisys_tensor.hpp"

#include "../../tensor/tensor.hpp"

#include "../../utils.hpp"

#include <random>
#include <vector>
#include <algorithm>
#include <cmath>

#include "../../ops/add/op.hpp"
#include "../../ops/argmax/op.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rearrange/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/self_attention/op.hpp"
#include "../../ops/swiglu/op.hpp"

//实现Qwen2的api

__C{
    struct LlaisysQwen2Model{
        LlaisysQwen2Meta meta;
        LlaisysQwen2Weights* weights;

        llaisysDeviceType_t device;
        int *device_ids;
        int ndevice;

        //定义kv cache相关
        llaisysTensor_t *k_cache;
        llaisysTensor_t *v_cache;

        // size_t cached = 0;
    };

    __export struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, const size_t kv_cache_size,
        llaisysDeviceType_t device, int *device_ids, int ndevice){
        //初始化模型
        LOG("llaisysQwen2ModelCreate");
        LlaisysQwen2Model *model = new LlaisysQwen2Model();
        model->meta = *meta;
        model->device = device;
        //深拷贝device_ids
        model->device_ids = new int[ndevice];
        std::copy(device_ids, device_ids + ndevice, model->device_ids);
        model->ndevice = ndevice;
        
        //分配weights结构体
        model->weights = new LlaisysQwen2Weights();

        //初始化kv cache
        size_t nlayer = meta->nlayer;
        model->k_cache = new llaisysTensor_t[nlayer];
        model->v_cache = new llaisysTensor_t[nlayer];
        size_t kv_shape[3] = {kv_cache_size, meta->nkvh, meta->dh};
        for(size_t i = 0; i < nlayer; i++){
            model->k_cache[i] = tensorCreate(kv_shape, 3, meta->dtype, device, device_ids[0]);
            model->v_cache[i] = tensorCreate(kv_shape, 3, meta->dtype, device, device_ids[0]);
        }
        LOG("llaisysQwen2ModelCreate: k_cache");
        LOG("llaisysQwen2ModelCreate: v_cache");


        //初始化其他权重
        LOG("llaisysQwen2ModelCreate: in_embed");
        std::vector<size_t> in_embed_shape = {meta->voc, meta->hs};
        model->weights->in_embed = tensorCreate(in_embed_shape.data(), 2, meta->dtype, device, device_ids[0]);
        LOG("llaisysQwen2ModelCreate: out_embed");
        std::vector<size_t> out_embed_shape = {meta->voc, meta->hs};
        model->weights->out_embed = tensorCreate(out_embed_shape.data(), 2, meta->dtype, device, device_ids[0]);
        LOG("llaisysQwen2ModelCreate: out_norm_w");
        std::vector<size_t> out_norm_w_shape = {meta->hs};
        model->weights->out_norm_w = tensorCreate(out_norm_w_shape.data(), 1, meta->dtype, device, device_ids[0]);

        //初始化其他权重
        model->weights->attn_norm_w = new llaisysTensor_t[nlayer];
        model->weights->attn_q_w = new llaisysTensor_t[nlayer];
        model->weights->attn_k_w = new llaisysTensor_t[nlayer];
        model->weights->attn_v_w = new llaisysTensor_t[nlayer];
        model->weights->attn_o_w = new llaisysTensor_t[nlayer];
        model->weights->attn_q_b = new llaisysTensor_t[nlayer];
        model->weights->attn_k_b = new llaisysTensor_t[nlayer];
        model->weights->attn_v_b = new llaisysTensor_t[nlayer];
        model->weights->mlp_up_w = new llaisysTensor_t[nlayer];
        model->weights->mlp_down_w = new llaisysTensor_t[nlayer];
        model->weights->mlp_gate_w = new llaisysTensor_t[nlayer];
        model->weights->mlp_norm_w = new llaisysTensor_t[nlayer];

        size_t attn_norm_w_shape[1] = {meta->hs};
        //hs = nh * dh
        size_t attn_q_w_shape[2] = {meta->hs, meta->hs};
        size_t attn_k_w_shape[2] = {meta->nkvh * meta->dh, meta->hs};
        size_t attn_v_w_shape[2] = {meta->nkvh * meta->dh, meta->hs};
        size_t attn_o_w_shape[2] = {meta->hs, meta->hs};

        size_t attn_q_b_shape[1] = {meta->hs};
        size_t attn_k_b_shape[1] = {meta->nkvh * meta->dh};
        size_t attn_v_b_shape[1] = {meta->nkvh * meta->dh};

        size_t mlp_down_w_shape[2] = {meta->hs, meta->di};
        size_t mlp_up_w_shape[2] = {meta->di, meta->hs};
        size_t mlp_gate_w_shape[2] = {meta->di, meta->hs};
        size_t mlp_norm_w_shape[1] = {meta->hs};

        for(size_t i = 0; i < nlayer; i++){
            model->weights->attn_norm_w[i] = tensorCreate(attn_norm_w_shape, 1, meta->dtype, device, device_ids[0]);
            model->weights->attn_q_w[i] = tensorCreate(attn_q_w_shape, 2, meta->dtype, device, device_ids[0]);
            model->weights->attn_k_w[i] = tensorCreate(attn_k_w_shape, 2, meta->dtype, device, device_ids[0]);
            model->weights->attn_v_w[i] = tensorCreate(attn_v_w_shape, 2, meta->dtype, device, device_ids[0]);
            model->weights->attn_o_w[i] = tensorCreate(attn_o_w_shape, 2, meta->dtype, device, device_ids[0]);
            model->weights->attn_q_b[i] = tensorCreate(attn_q_b_shape, 1, meta->dtype, device, device_ids[0]);
            model->weights->attn_k_b[i] = tensorCreate(attn_k_b_shape, 1, meta->dtype, device, device_ids[0]);
            model->weights->attn_v_b[i] = tensorCreate(attn_v_b_shape, 1, meta->dtype, device, device_ids[0]);
            model->weights->mlp_down_w[i] = tensorCreate(mlp_down_w_shape, 2, meta->dtype, device, device_ids[0]);
            model->weights->mlp_up_w[i] = tensorCreate(mlp_up_w_shape, 2, meta->dtype, device, device_ids[0]);
            model->weights->mlp_gate_w[i] = tensorCreate(mlp_gate_w_shape, 2, meta->dtype, device, device_ids[0]);
            model->weights->mlp_norm_w[i] = tensorCreate(mlp_norm_w_shape, 1, meta->dtype, device, device_ids[0]);
        }
        LOG("model created");

        return model;
    }

    __export void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model * model){
        if(model == nullptr){
            return;
        }

        size_t nlayer = model->meta.nlayer;
        //销毁kv cache
        for(size_t i = 0; i < nlayer; i++){
            tensorDestroy(model->k_cache[i]);
            tensorDestroy(model->v_cache[i]);
        }
        delete[] model->k_cache;
        delete[] model->v_cache;
        
        //销毁其他权重
        tensorDestroy(model->weights->in_embed);
        tensorDestroy(model->weights->out_embed);
        tensorDestroy(model->weights->out_norm_w);
        for(size_t i = 0; i < nlayer; i++){
            tensorDestroy(model->weights->attn_norm_w[i]);
            tensorDestroy(model->weights->attn_q_w[i]);
            tensorDestroy(model->weights->attn_k_w[i]);
            tensorDestroy(model->weights->attn_v_w[i]);
            tensorDestroy(model->weights->attn_o_w[i]);
            tensorDestroy(model->weights->attn_q_b[i]);
            tensorDestroy(model->weights->attn_k_b[i]);
            tensorDestroy(model->weights->attn_v_b[i]);
            tensorDestroy(model->weights->mlp_down_w[i]);
            tensorDestroy(model->weights->mlp_up_w[i]);
            tensorDestroy(model->weights->mlp_gate_w[i]);
            tensorDestroy(model->weights->mlp_norm_w[i]);
        }
        //销毁其他权重
        delete[] model->weights->attn_norm_w;
        delete[] model->weights->attn_q_w;
        delete[] model->weights->attn_k_w;
        delete[] model->weights->attn_v_w;
        delete[] model->weights->attn_o_w;
        delete[] model->weights->attn_q_b;
        delete[] model->weights->attn_k_b;
        delete[] model->weights->attn_v_b;
        delete[] model->weights->mlp_down_w;
        delete[] model->weights->mlp_up_w;
        delete[] model->weights->mlp_gate_w;
        delete[] model->weights->mlp_norm_w;
        
        //销毁weights结构体
        delete model->weights;

        if(model -> device_ids != nullptr){
            delete[] model -> device_ids;
        }
        delete model;
    }

    __export struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model * model){
        // TO_BE_IMPLEMENTED();
        if(model != nullptr){
            return model->weights;
        }
        return nullptr;
    }

    __export int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model, int64_t * token_ids, size_t ntoken, size_t past_len){
        /*
            reference:https://github.com/zebra-uestc/llaisys/blob/master/src/llaisys/models/qwen2.cc\
        */
        // TO_BE_IMPLEMENTED();
        if(model == nullptr || token_ids == nullptr || ntoken == 0){
            std::cerr << "model or token_ids is nullptr" << std::endl;
            return -1;
        }
        
        //weights
        if(model -> weights->in_embed == nullptr || 
            model -> weights->out_embed == nullptr ||
            model -> weights->out_norm_w == nullptr){
            std::cerr << "in_embed or out_embed or out_norm_w is nullptr" << std::endl;
            return -1;
        }

        //layer weights
        int device_id = 0;
        if(model -> device_ids != nullptr && model -> ndevice > 0){
            device_id = model -> device_ids[0];
        }
        auto device_type = model->device;

        //some variables
        size_t nlayer = model->meta.nlayer;
        size_t voc = model->meta.voc;
        size_t di = model->meta.di;
        size_t dh = model->meta.dh;
        size_t nh = model->meta.nh;
        size_t hs = model->meta.hs;
        size_t nkvh = model->meta.nkvh;
        size_t dkvh = dh * nkvh;

        float eps = model->meta.epsilon;
        float theta = model->meta.theta;
        float scale = 1.0f / std::sqrt(static_cast<float>(dh));

        size_t seqlen = ntoken;
        size_t curlen = past_len + seqlen;

        auto k_cache = model->k_cache;
        auto v_cache = model->v_cache;

        std::vector<size_t> create_shape = {seqlen};
        //pos_ids for rope
        auto pos_ids = tensorCreate(create_shape.data(), 1, LLAISYS_DTYPE_I64, device_type, device_id);
        //fill pos_ids
        auto pos_ids_data = reinterpret_cast<int64_t *>(tensorGetData(pos_ids));
        for(size_t i = 0; i < seqlen; i++){
            pos_ids_data[i] = static_cast<int64_t>(i + past_len);
        }

        //load token_ids
        auto input_token_ids = tensorCreate(create_shape.data(), 1, LLAISYS_DTYPE_I64, device_type, device_id);
        tensorLoad(input_token_ids, token_ids);

        //input embed
        create_shape = {seqlen, hs};
        auto input_embed = tensorCreate(create_shape.data(), 2, model->meta.dtype, device_type, device_id);
        llaisysEmbedding(input_embed, input_token_ids, model->weights->in_embed);


        //transformer
        for(size_t i = 0; i < nlayer; i++){
            LOG(std::string("layer ") + std::to_string(i) + "/" + std::to_string(nlayer -1));
            /*
                normalization
            */
            auto attn_norm = tensorCreate(create_shape.data(), 2, model->meta.dtype, device_type, device_id);
            llaisysRmsNorm(attn_norm, input_embed, model->weights->attn_norm_w[i], eps);
            // LOG("attn_norm");
            // tensorDebug(attn_norm);

            /*
                attention 
            */
            //q_proj
            auto q_proj = tensorCreate(create_shape.data(), 2, model->meta.dtype, device_type, device_id);
            llaisysLinear(q_proj, attn_norm, model->weights->attn_q_w[i], model->weights->attn_q_b[i]);
            // LOG("q_proj");
            // tensorDebug(q_proj);

            //q_rope
            create_shape = {seqlen, nh, dh};
            q_proj = tensorView(q_proj, create_shape.data(), 3);
            auto q_rope = tensorCreate(create_shape.data(), 3, model->meta.dtype, device_type, device_id);
            llaisysROPE(q_rope, q_proj, pos_ids, theta);
            // LOG("q_rope");
            // tensorDebug(q_rope);

            //k_proj
            create_shape = {seqlen, dkvh};
            auto k_proj = tensorCreate(create_shape.data(), 2, model->meta.dtype, device_type, device_id);
            llaisysLinear(k_proj, attn_norm, model->weights->attn_k_w[i], model->weights->attn_k_b[i]);
            // LOG("k_proj");
            // tensorDebug(k_proj);

            //k_rope
            create_shape = {seqlen, nkvh, dh};
             k_proj = tensorView(k_proj, create_shape.data(), 3);
            auto k_rope = tensorSlice(k_cache[i], 0, past_len, curlen);
            llaisysROPE(k_rope, k_proj, pos_ids, theta);
            // LOG("k_rope");
            // tensorDebug(k_rope);

            //v_proj in_place
            create_shape = {seqlen, dkvh};
            auto v_proj = tensorView(tensorSlice(v_cache[i], 0, past_len, curlen), create_shape.data(), 2);
            llaisysLinear(v_proj, attn_norm, model->weights->attn_v_w[i], model->weights->attn_v_b[i]);
            // LOG("v_proj");
            // tensorDebug(v_proj);

            //self attention
            create_shape = {seqlen, nh, dh};
            auto attn_val = tensorCreate(create_shape.data(), 3, model->meta.dtype, device_type, device_id);

            auto attn_k = tensorSlice(k_cache[i], 0, 0, curlen);
            auto attn_v = tensorSlice(v_cache[i], 0, 0, curlen);

            llaisysSelfAttention(attn_val, q_rope, attn_k, attn_v, scale);
            // LOG("self_attention");
            // tensorDebug(attn_out);


//o_proj
             create_shape = {seqlen, hs};
            auto o_proj = tensorCreate(create_shape.data(), 2, model->meta.dtype, device_type, device_id);
            attn_val = tensorView(attn_val, create_shape.data(), 2);
            // LOG("attn_val");
            // tensorDebug(attn_val);
            llaisysLinear(o_proj, attn_val, model->weights->attn_o_w[i], nullptr);
            // LOG("o_proj");
            // tensorDebug(o_proj);

            //residual connection
            create_shape = {seqlen, hs};
            auto mlp_layer = tensorCreate(create_shape.data(), 2, model->meta.dtype, device_type, device_id);
            llaisysAdd(mlp_layer, input_embed, o_proj);
            // LOG("mlp_layer");

            //mlp_norm
            create_shape = {seqlen, hs};
            auto mlp_norm = tensorCreate(create_shape.data(), 2, model->meta.dtype, device_type, device_id);
            llaisysRmsNorm(mlp_norm, mlp_layer, model->weights->mlp_norm_w[i], eps);
            // LOG("mlp_norm");
            // tensorDebug(mlp_norm);

            /*
                mlp
            */

            //mlp_gate
            create_shape = {seqlen, di};
            auto mlp_gate = tensorCreate(create_shape.data(), 2, model->meta.dtype, device_type, device_id);
            llaisysLinear(mlp_gate, mlp_norm, model->weights->mlp_gate_w[i], nullptr);
            // LOG("mlp_gate");
            // tensorDebug(mlp_gate);

             //mlp_up
            create_shape = {seqlen, di};
            auto mlp_up = tensorCreate(create_shape.data(), 2, model->meta.dtype, device_type, device_id);
            llaisysLinear(mlp_up, mlp_norm, model->weights->mlp_up_w[i], nullptr);
            // LOG("mlp_up");
            // tensorDebug(mlp_up);

             //mlp_swiglu
            create_shape = {seqlen, di};
            auto mlp_swiglu = tensorCreate(create_shape.data(), 2, model->meta.dtype, device_type, device_id);
            llaisysSwiGLU(mlp_swiglu, mlp_gate, mlp_up);
            // LOG("mlp_swiglu");
            // tensorDebug(mlp_swiglu);

//mlp_down
             create_shape = {seqlen, hs};
            auto mlp_down = tensorCreate(create_shape.data(), 2, model->meta.dtype, device_type, device_id);
            llaisysLinear(mlp_down, mlp_swiglu, model->weights->mlp_down_w[i], nullptr);
            // LOG("mlp_down");
            // tensorDebug(mlp_down);

            //residual connection
             llaisysAdd(input_embed, mlp_layer, mlp_down);
            // LOG("residual connection");
            // tensorDebug(mlp_layer);

            tensorDestroy(attn_norm);
            tensorDestroy(q_proj);
            tensorDestroy(k_proj);
            tensorDestroy(v_proj);
            tensorDestroy(attn_val);
            tensorDestroy(o_proj);
            tensorDestroy(mlp_layer);
            tensorDestroy(mlp_norm);
            tensorDestroy(mlp_gate);
            tensorDestroy(mlp_up);
            tensorDestroy(mlp_swiglu);
            tensorDestroy(mlp_down);
        }

        /*
            output
        */
        LOG("It is time to output");
        //rms_norm
        create_shape = {seqlen, hs};
        auto output = tensorCreate(create_shape.data(), 2, model->meta.dtype, device_type, device_id);
        llaisysRmsNorm(output, input_embed, model->weights->out_norm_w, eps);
        // tensorDebug(output);

        
        //output_embed
        create_shape = {seqlen, voc};
        auto output_embed = tensorCreate(create_shape.data(), 2, model->meta.dtype, device_type, device_id);
        llaisysLinear(output_embed, output, model->weights->out_embed, nullptr);
        // tensorDebug(output_embed);

//argmax
        auto output_ids = tensorSlice(output_embed, 0, seqlen - 1, seqlen);

        create_shape = {1};
        auto max_val = tensorCreate(create_shape.data(), 1, model->meta.dtype, device_type, device_id);
        auto max_ids = tensorCreate(create_shape.data(), 1, LLAISYS_DTYPE_I64, device_type, device_id);
        llaisysArgmax(max_ids, max_val, output_ids);

        int64_t token_id = *reinterpret_cast<int64_t *>(tensorGetData(max_ids));       

        //destroy
        tensorDestroy(pos_ids);
        tensorDestroy(input_token_ids);
        tensorDestroy(input_embed);
        tensorDestroy(output);
        tensorDestroy(output_embed);
        tensorDestroy(max_val);
        tensorDestroy(max_ids);

        return token_id;
    }
}

