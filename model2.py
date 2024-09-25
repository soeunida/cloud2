from typing import List, Optional, Tuple, Union
from dataclasses import dataclass
from collections import defaultdict
import torch
import torch.utils.checkpoint
from torch import nn
from transformers import PreTrainedModel
import gc
from transformers import Phi3Config
from transformers.utils import ModelOutput
from safetensors import safe_open
import json
from hf_ref import (
    _prepare_4d_causal_attention_mask_with_cache_position,
    Phi3RMSNorm,
    Phi3DecoderLayer,
    NewPhi3Config
)

from transformers.cache_utils import StaticCache
import os
from transformers.utils import ModelOutput

device = torch.device('cuda:2')
torch.cuda.set_device(device) 
pre_weight_map = {}
file_num = 1
tensor_dict = {}

def load_one_file():
    
    global pre_weight_map, tensor_dict, file_num
    
    if file_num > 6:
        print("파일 번호가 6번을 넘어감")
    file_path = f'/nas/user/hayoung/model-0000{file_num}-of-00006.safetensors'
    
    with safe_open(file_path, framework="pt", device="cuda") as f:
        for key in f.keys():
            tensor_dict[key] = f.get_tensor(key)
    
    file_num += 1
    
    
def save_index_dict():
    """
    여기 바꾸기 -> 다운로드 path에 뒤에 붙이기
    """
    cur_dir = '/nas/user/hayoung'
    #idx_file_folder = cur_dir + '/models--microsoft--Phi-3-medium-4k-instruct/snapshots/d194e4e74ffad5a5e193e26af25bcfc80c7f1ffc'
    idx_file_path = cur_dir + '/' + 'model.safetensors.index.json'

    global pre_weight_map
    with open(idx_file_path, 'r', encoding='utf-8') as f:
        data_dict = json.load(f)
        pre_weight_map = data_dict['weight_map'].copy()
    

def _prepare_4d_causal_attention_mask_with_cache_position(
    attention_mask: torch.Tensor,
    sequence_length: int,
    target_length: int,
    dtype: torch.dtype,
    device: torch.device,
    min_dtype: float,
    cache_position: torch.Tensor,
    batch_size: int,
):
    if attention_mask is not None and attention_mask.dim() == 4:
        # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
        causal_mask = attention_mask
    else:
        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )

    return causal_mask
@dataclass
class CausalLMOutputWithPast(ModelOutput):
    logits: torch.FloatTensor = None

class Phi3PreTrainedModel(PreTrainedModel):
    config_class = NewPhi3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Phi3DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    _version = "0.0.5"

                
class EmbedModel(nn.Module):
    def __init__(self, config: Phi3Config):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.base_path = config.base_path
        global device
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx).to(device)
        self.embed_dropout = nn.Dropout(config.embd_pdrop)   
        
    def load_weights(self):
        global pre_weight_map, tensor_dict
        
        new_state_dict = {}
        new_state_dict['embed_tokens.weight'] = tensor_dict['model.embed_tokens.weight']
        
        self.load_state_dict(new_state_dict)
        del tensor_dict['model.embed_tokens.weight']
    
    def forward(
        self,
        input_ids: torch.LongTensor = None
    ):
        
        self.load_weights()
        inputs_embeds = self.embed_tokens(input_ids)

        return inputs_embeds
        
        
        
        
class Body(Phi3PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Phi3DecoderLayer`]

    Args:
        config: Phi3Config
    """

    def __init__(self, block_size, config: NewPhi3Config):
        super().__init__(config)
        global device
        self.layers = nn.ModuleList(
            [Phi3DecoderLayer(config, i) for i in range(block_size)]
        ).to(device)
        self._attn_implementation = config._attn_implementation
        self.base_path = config.base_path
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.block_size = block_size


    def load_weights(self, idx):
        
        new_state_dict = {}
        global pre_weight_map, tensor_dict
        
        partial_model_keys = list(self.layers.state_dict().keys())
        
        names = []
        for key in partial_model_keys:
            try:
                num = int(key[0:2])
                pre_num = num + idx
                pre_name = "model.layers." + str(pre_num) + key[2:]
            except:
                num = int(key[0])
                pre_num = num + idx
                pre_name = "model.layers." + str(pre_num) + key[1:]
            try:
                new_state_dict[key] = tensor_dict[pre_name]
                names.append(pre_name)
            except:
                load_one_file()
                new_state_dict[key] = tensor_dict[pre_name]
        
        
        self.layers.load_state_dict(new_state_dict)
        
        for name in names:
            del tensor_dict[name]
        del partial_model_keys
        
        
        


    def forward(
        self,
        idx,
        hidden_states: torch.LongTensor = None,
        causal_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        

        for decoder_layer in self.layers:
            
            layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    cache_position=cache_position,
                    output_attentions=None,
                    use_cache=False
                )

            hidden_states = layer_outputs[0]

        return hidden_states


class CustomedPhi3ForCausalLM(Phi3PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.__init__ with Llama->Phi3
    def __init__(self, config):
        super().__init__(config)
        global device
        
        self.norm = Phi3RMSNorm(config.hidden_size).to(device)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False).to(device)
        self.config = config
        self.base_path = config.base_path

    def load_weights(self):
        global pre_weight_map, tensor_dict
        new_state_dict = {}
 
        new_state_dict['norm.weight'] = tensor_dict['model.norm.weight']
        new_state_dict['lm_head.weight'] = tensor_dict['lm_head.weight']
                    
        self.load_state_dict(new_state_dict)
        
        del tensor_dict['model.norm.weight']
        del tensor_dict['lm_head.weight']
        
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        global file_num
        save_index_dict()
        file_num = 1
        load_one_file()
        torch.cuda.nvtx.range_push("embed model load")  
        embed_model = EmbedModel(self.config)
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("embed forward")   
        hidden_states = embed_model(input_ids)
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("del embed model")
        del embed_model
        torch.cuda.nvtx.range_pop()

        past_seen_tokens = 0
        cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + hidden_states.shape[1], device=device
            )
        position_ids = cache_position.unsqueeze(0)
        
        causal_mask = attention_mask

        stream1 = torch.cuda.Stream()
        stream2 = torch.cuda.Stream()
        
        with torch.cuda.stream(stream1):
            body1 = Body(self.config.block_size, self.config)
            body1.load_weights(0)
            
        with torch.cuda.stream(stream2):
            body2 = Body(self.config.block_size, self.config)
            
        for idx in range(0, 40, self.config.block_size*2):
            with torch.cuda.stream(stream1):
                hidden_states = body1(idx, hidden_states, causal_mask, position_ids, None, cache_position)
                #print(f'stream 1 이 {idx} 블럭 계산중')
            with torch.cuda.stream(stream2):
                body2.load_weights(idx + self.config.block_size)
                #print(f'stream 2 이 {idx+self.config.block_size} 블럭 load 중')
            torch.cuda.synchronize()
            
            with torch.cuda.stream(stream1):
                if (idx + self.config.block_size*2) == 40:
                    break
                body1.load_weights(idx + self.config.block_size*2)
            with torch.cuda.stream(stream2):
                hidden_states = body2(idx + self.config.block_size, hidden_states, causal_mask, position_ids, None, cache_position)
            torch.cuda.synchronize()
            
        del body1
        del body2

        hidden_states = hidden_states
        self.load_weights()
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        print('forward')

        return CausalLMOutputWithPast(
            logits=logits
        )

    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        **kwargs,
    ):
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

                # This `clone` call is needed to avoid recapturing cuda graphs with `torch.compile`'s  `mode="reduce-overhead`, as otherwise the input `position_ids` would have various stride during the decoding. Here, simply using `.contiguous()` is not sufficient as in the batch size = 1 case, `position_ids` is already contiguous but with varying stride which retriggers a capture.
                position_ids = position_ids.clone(memory_format=torch.contiguous_format)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            # The clone here is for the same reason as for `position_ids`.
            model_inputs = {"input_ids": input_ids.clone(memory_format=torch.contiguous_format), "inputs_embeds": None}

        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
                device = model_inputs["inputs_embeds"].device
            else:
                batch_size, sequence_length = model_inputs["input_ids"].shape
                device = model_inputs["input_ids"].device

            dtype = self.lm_head.weight.dtype
            min_dtype = torch.finfo(dtype).min

            attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.get_max_length(),
                dtype=dtype,
                device=device,
                min_dtype=min_dtype,
                cache_position=cache_position,
                batch_size=batch_size,
            )

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs
    
    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values =None,
        output_attentions= False,
    ):

        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens =  0

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        target_length = (
                attention_mask.shape[-1]
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = _prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            min_dtype=min_dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )


        return causal_mask
    
    

#     # Install torch-tensorrt
# # !pip install torch-tensorrt -f https://github.com/pytorch/TensorRT/releases
# from customed_pipeline import CustomedPipeline
# import torch
# import onnx
# import torch_tensorrt as trt
# import tensorrt as trt_rt
# import pycuda.driver as cuda
# import pycuda.autoinit  # Initializes CUDA driver
# from model2 import CustomedPhi3ForCausalLM
# from transformers import AutoTokenizer
# from hf_ref import NewPhi3Config
# import requests

# # Function to download model files
# def download_model():
#     base_path = '/mnt/sd/phi3/'
#     for i in range(6):
#         file_path = base_path + f'model-0000{i+1}-of-00006.safetensors'
#         with open(file_path, 'wb') as device_file:
#             path = f'https://huggingface.co/microsoft/Phi-3-medium-4k-instruct/resolve/main/model-0000{i+1}-of-00006.safetensors'
#             response = requests.get(path, stream=True)
#             print(f'{i+1}번째 파일 status ', response.status_code)
#             if response.status_code == 200:
#                 for chunk in response.iter_content(chunk_size=1024 * 1024):  
#                     if chunk: 
#                         device_file.write(chunk)

# download_model()

# # Initialize device and set seed
# torch.random.manual_seed(0)
# device = torch.device('cuda:2')
# torch.cuda.set_device(device) 

# # Load model and tokenizer
# model_id = "microsoft/Phi-3-medium-4k-instruct"
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# config = NewPhi3Config()
# model = CustomedPhi3ForCausalLM(config)

# # Set the model to evaluation mode
# model.eval()

# # Create dummy input for the model
# dummy_input_ids = torch.randint(0, config.vocab_size, (1, 10)).to(device)  # Adjust the shape accordingly
# dummy_attention_mask = torch.ones((1, 10)).to(device)  # Attention mask for the dummy input

# # Export the model to ONNX format
# onnx_model_path = "customed_phi3_model.onnx"
# torch.onnx.export(
#     model,
#     (dummy_input_ids, dummy_attention_mask),
#     onnx_model_path,
#     input_names=["input_ids", "attention_mask"],
#     output_names=["logits"],
#     opset_version=13,  # Use a compatible ONNX opset version
#     dynamic_axes={
#         "input_ids": {0: "batch_size", 1: "sequence_length"},
#         "attention_mask": {0: "batch_size", 1: "sequence_length"},
#         "logits": {0: "batch_size", 1: "sequence_length"}
#     }
# )
# print("Model has been exported to ONNX format.")

# # Convert ONNX model to TensorRT using torch_tensorrt
# trt_model = trt.compile(
#     module=onnx_model_path,
#     inputs=[trt.Input((1, 10))],  # Adjust the input size accordingly
#     enabled_precisions={torch.float, torch.half},  # Use FP32 or FP16
#     truncate_long_and_flat_inputs=False
# )

# # Save the converted TensorRT model
# trt_model_path = "customed_phi3_model_trt.engine"
# with open(trt_model_path, "wb") as f:
#     f.write(trt_model)
# print("Model has been converted to TensorRT format and saved.")

# # Load TensorRT engine and run inference
# def load_trt_engine(engine_file_path):
#     TRT_LOGGER = trt_rt.Logger(trt_rt.Logger.WARNING)
#     with open(engine_file_path, "rb") as f:
#         engine_data = f.read()
#     runtime = trt_rt.Runtime(TRT_LOGGER)
#     engine = runtime.deserialize_cuda_engine(engine_data)
#     return engine, runtime

# # Allocate device memory and perform TensorRT inference
# def infer_trt(engine, input_tensor):
#     context = engine.create_execution_context()

#     input_shape = input_tensor.shape
#     input_size = input_tensor.nelement() * input_tensor.element_size()

#     # Allocate memory on the device for input/output
#     d_input = cuda.mem_alloc(input_size)
#     d_output = cuda.mem_alloc(input_size)

#     bindings = [int(d_input), int(d_output)]

#     # Transfer input data to the device
#     cuda.memcpy_htod(d_input, input_tensor.cpu().numpy())

#     # Execute inference
#     context.execute_v2(bindings)

#     # Transfer output data back to the host
#     output_tensor = torch.empty(input_shape).to("cuda")
#     cuda.memcpy_dtoh(output_tensor.cpu().numpy(), d_output)

#     return output_tensor

# # Initialize the pipeline
# pipe = CustomedPipeline(model, config)
# pipe.load_data(dataset_name="allenai/openbookqa", split='validation', batch_size=40)

# # Example usage of the inference pipeline
# engine_file_path = "customed_phi3_model_trt.engine"
# engine, runtime = load_trt_engine(engine_file_path)

# # Forward pass using the pipeline
# outputs = pipe.forward(max_new_tokens=15)

# # Assuming pipe.forward() gives batches with "input_ids"
# for batch in outputs:
#     input_tensor = batch["input_ids"].to("cuda")  # Adjust this based on pipeline outputs
#     trt_outputs = infer_trt(engine, input_tensor)

#     # Postprocess the outputs
#     result = pipe.postprocess(trt_outputs)
#     print(result)
