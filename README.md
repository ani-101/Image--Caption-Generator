# Image--Caption-Generator
This project is a prototype of Image-caption generator, using Transformer

# Dataset
This caption generartor can recognize only dogs and child(age from 3 to 13) images 
This dataset contains 50 images of dogs and 52 images are of toddlers and children
Each image has its five corresponding captions
The images are of high quality(more than 224x224 resolution)

# For creating TensorFlow Datasets to be used in training the model
In this, we created a TensorFlow Dataset using the 'from_tensor_slices' method, which takes a tuple of two tensors, 'train_imgs' and 'train_captions', as input. Each element of these tensors corresponds to an image and its corresponding caption.

The next line applies the 'map' function to the 'train_dataset' created above. The 'load_data' function is applied to each element of the dataset in parallel using the 'num_parallel_calls' argument set to 'tf.data.AUTOTUNE'. This is a technique to optimize the data loading process by automatically tuning the level of parallelism based on the available computational resources.

The resulting dataset is then shuffled using the 'shuffle' function with a buffer size specified by the 'BUFFER_SIZE' parameter. The 'batch' function is then used to batch the dataset into groups of size specified by the 'BATCH_SIZE' parameter.

The same process is then repeated for the validation dataset with the 'val_dataset' variable created in a similar manner to 'train_dataset'.

# Working
In this, we used the InceptionV3 architecture that is pre-trained on the ImageNet dataset to extract image features for an image captioning task.

The output from the InceptionV3 model is then reshaped using a Keras Reshape layer to create a 3D tensor of shape (num_regions, num_features) where 'num_regions' is the number of spatial regions in the input image and 'num_features' is the number of features in each region.

Finally, a Keras Model is created that takes the input tensor from the InceptionV3 model and outputs the reshaped tensor. This model will be used to encode the input images and extract their feature representations, which can then be used as input to a decoder model that generates captions for the images.

The TransformerEncoderLayer class defines a single Transformer encoder layer, with multi-head attention and a feedforward neural network (FFN) for each head.

The Embeddings class defines the embedding layer that combines token embeddings with positional embeddings.

The TransformerDecoderLayer class defines a single Transformer decoder layer, with two multi-head attention blocks - one for attending over the encoder output and another for self-attention - and an FFN block.

The ImageCaptioningModel class defines the entire image captioning model. It takes in the CNN Encoder, Transformer Encoder, and Transformer Decoder as arguments, along with an optional image augmentation layer. It defines the forward pass for the entire model, which first passes the input image through the CNN Encoder to obtain image features, then passes the image features through the Transformer Encoder, and finally passes the resulting encoder output through the Transformer Decoder to generate a caption.

The loss_tracker attribute of the ImageCaptioningModel class tracks the loss during training.
