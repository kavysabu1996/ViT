import numpy as np
from linear_layer import Linear
import tensorflow as tf
import typing

class ViT(tf.keras.Model):
    def __init__(self,patch_size=16,num_layers=12,model_dim=768,num_heads=12):
        super().__init__()
        self.patch_size = patch_size 
        self.num_layers = num_layers
        self.model_dim = model_dim
        self.num_heads = num_heads

    def build(self,input_shapes):
        self.embedding= tf.keras.layers.Conv2D(filters=self.model_dim,kernel_size=self.patch_size,
                                    strides=self.patch_size,padding="valid",name="embedding")
        emb_w = np.load("weights/embedding/kernel.npy")
        emb_b = np.load("weights/embedding/bias.npy")
        tmp = self.embedding(tf.zeros(shape=input_shapes))
        self.embedding.set_weights([emb_w,emb_b])
        self.class_token = np.load("weights/class_token/cls.npy")
        self.pos_emb = np.load("weights/Transformer/posembed_input/pos_embedding.npy")

        self.encoder_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="Transformer/encoder_norm")
        enc_norm_gamma = np.load("weights/Transformer/encoder_norm/gamma.npy")
        enc_norm_beta = np.load("weights/Transformer/encoder_norm/beta.npy")
        tmp = self.encoder_norm(tf.zeros(tmp.shape))
        self.encoder_norm.set_weights([enc_norm_gamma,enc_norm_beta]) 

        head_w = np.load("weights/head/kernel.npy")
        head_b = np.load("weights/head/bias.npy")
        self.head = tf.keras.layers.Dense(1000, name="head", activation="sigmoid")
        tmp = self.head(tf.zeros(shape=(1,1,768)))
        self.head.set_weights([head_w,head_b])

    def call(self, inputs):             
        y = self.embedding(inputs)
        y = tf.keras.layers.Reshape((y.shape[1] * y.shape[2], self.model_dim))(y)
        y = tf.concat([self.class_token, y], 1)
        y = tf.keras.layers.Add()([y,self.pos_emb])[np.newaxis,...]
        for n in range(self.num_layers):
            encoder = EncoderLayer(layer_name=f"encoderblock_{n}")
            output = encoder(y)
            y = output                          
        y = self.encoder_norm(y)        
        y = tf.keras.layers.Lambda(lambda v: v[:,:,0], name="ExtractToken")(y)
        y = self.head(y)
        return y
        
class EncoderLayer(tf.keras.Model):
    def __init__(self, layer_name,num_heads=12, mlp_dim=3072, model_dim=768):
        super().__init__()
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.model_dim = model_dim
        self.layer_name = layer_name
    
    def build(self,input_shapes):
        self.attn = MultiHeadAttention(self.num_heads,self.model_dim,self.layer_name)

        Dense0_kernel = np.squeeze(np.load("weights/Transformer/{}/Dense_0/kernel.npy".format(self.layer_name)))[np.newaxis,np.newaxis,...]
        Dense1_kernel = np.squeeze(np.load("weights/Transformer/{}/Dense_1/kernel.npy".format(self.layer_name)))[np.newaxis,np.newaxis,...]
        Dense0_bias = np.load("weights/Transformer/{}/Dense_0/bias.npy".format(self.layer_name))
        Dense1_bias = np.load("weights/Transformer/{}/Dense_1/bias.npy".format(self.layer_name))
        self.Dense0 = Linear(Dense0_kernel,Dense0_bias,"{}_Dense0".format(self.layer_name))
        self.Dense1 = Linear(Dense1_kernel,Dense1_bias,"{}_Dense1".format(self.layer_name))

        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="LayerNorm_1")
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="LayerNorm_2")
        norm1_gamma = np.load("weights/Transformer/{}/LayerNorm_0/gamma.npy".format(self.layer_name))
        norm2_gamma = np.load("weights/Transformer/{}/LayerNorm_2/gamma.npy".format(self.layer_name))
        norm1_beta = np.load("weights/Transformer/{}/LayerNorm_0/beta.npy".format(self.layer_name))
        norm2_beta = np.load("weights/Transformer/{}/LayerNorm_2/beta.npy".format(self.layer_name))

        tmp1 = self.norm1(tf.zeros(input_shapes))
        tmp2 = self.norm2(tf.zeros(input_shapes))
        self.norm1.set_weights([norm1_gamma,norm1_beta])
        self.norm2.set_weights([norm2_gamma,norm2_beta])
        self.lambda1 = tf.keras.layers.Lambda(lambda x: tf.keras.activations.gelu(x, approximate=False))
    
    def call(self,inputs):        
        x = self.norm1(inputs)
        x = self.attn(x)
        x = tf.keras.layers.Add()([x,inputs])
        y = self.norm2(x)
        y = self.Dense0(y)
        y = self.lambda1(y)
        y = self.Dense1(y)
        return x+y

class MultiHeadAttention(tf.keras.Model):
    def __init__(self,num_heads,model_dim,layer_name):
        super().__init__()
        self.layer_name = layer_name
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.head_dim = model_dim//num_heads

    def build(self,input_shapes):
        wt_path = "weights/Transformer/{}/MultiHeadDotProductAttention_1/".format(self.layer_name)
        wq = np.squeeze(np.load(wt_path + "query/kernel.npy"))[np.newaxis,np.newaxis,...]
        wk = np.squeeze(np.load(wt_path + "key/kernel.npy"))[np.newaxis,np.newaxis,...]
        wv = np.squeeze(np.load(wt_path + "value/kernel.npy"))[np.newaxis,np.newaxis,...]
        bq = np.load(wt_path + "query/bias.npy")
        bk = np.load(wt_path + "key/bias.npy")
        bv = np.load(wt_path + "value/bias.npy")
        out_proj_kernel = np.squeeze(np.load(wt_path +"out/kernel.npy"))[np.newaxis,np.newaxis,...]
        out_proj_bias = np.load(wt_path + "out/bias.npy")
        self.query_proj = Linear(wq,bq,"{}_query_proj".format(self.layer_name))
        self.key_proj = Linear(wk,bk,"{}_key_proj".format(self.layer_name))
        self.value_proj = Linear(wv,bv,"{}_value_proj".format(self.layer_name))
        self.out_proj = Linear(out_proj_kernel,out_proj_bias,"{}_out_proj".format(self.layer_name))
          
    def get_input_slices(self,inputs):
        slices = []
        num_patches = inputs.shape[-2]
        for num in range(self.num_heads):
            slice = tf.strided_slice(inputs, begin=[0,0,0,num*self.head_dim], end=[1,1,num_patches,(num+1)*self.head_dim])
            slices.append(slice)
        return slices
    
    def get_attention(self,query,key,value):
        score = tf.matmul(query, key, transpose_b=True)
        weights = tf.nn.softmax(score, axis=-1)
        attn_output =tf.matmul(weights, value)
        return attn_output
       
    def call(self, inputs):
        query_proj = self.query_proj(inputs)
        query_proj *= float(self.head_dim)** -0.5
        key_proj = self.key_proj(inputs)
        value_proj = self.value_proj(inputs)

        query_slices = self.get_input_slices(query_proj)
        key_slices = self.get_input_slices(key_proj)
        value_slices = self.get_input_slices(value_proj)   
        
        attn_slices = []
        for q,k,v in zip(query_slices,key_slices,value_slices):
            attn_slice = self.get_attention(q,k,v)
            attn_slices.append(attn_slice)

        attn_output = tf.keras.layers.concatenate(attn_slices, axis=-1)
        attn_output_proj = self.out_proj(attn_output)
        return attn_output_proj
