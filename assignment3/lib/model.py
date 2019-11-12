# inbuilt lib imports:
from typing import Dict
import math

# external libs
import tensorflow as tf
from tensorflow.keras import models, layers

# project imports


class CubicActivation(layers.Layer):
    """
    Cubic activation as described in the paper.
    """
    def call(self, vector: tf.Tensor) -> tf.Tensor:
        """
        Parameters
        ----------
        vector : ``tf.Tensor``
            hidden vector of dimension (batch_size, hidden_dim)

        Returns tensor after applying the activation
        """
        # TODO(Students) Start
        # Comment the next line after implementing call.
        #raise NotImplementedError

        # TODO(Students) End
        return tf.pow(vector,3)


class DependencyParser(models.Model):
    def __init__(self,
                 embedding_dim: int,
                 vocab_size: int,
                 num_tokens: int,
                 hidden_dim: int,
                 num_transitions: int,
                 regularization_lambda: float,
                 trainable_embeddings: bool,
                 activation_name: str = "cubic") -> None:
        """
        This model defines a transition-based dependency parser which makes
        use of a classifier powered by a neural network. The neural network
        accepts distributed representation inputs: dense, continuous
        representations of words, their part of speech tags, and the labels
        which connect words in a partial dependency parse.
 
        This is an implementation of the method described in

        Danqi Chen and Christopher Manning.
        A Fast and Accurate Dependency Parser Using Neural Networks. In EMNLP 2014.

        Parameters
        ----------
        embedding_dim : ``str``
            Dimension of word embeddings
        vocab_size : ``int``
            Number of words in the vocabulary.
        num_tokens : ``int``
            Number of tokens (words/pos) to be used for features
            for this configuration.
        hidden_dim : ``int``
            Hidden dimension of feedforward network
        num_transitions : ``int``
            Number of transitions to choose from.
        regularization_lambda : ``float``
            Regularization loss fraction lambda as given in paper.
        trainable_embeddings : `bool`
            Is the embedding matrix trainable or not.
        """
        super(DependencyParser, self).__init__()
        self._regularization_lambda = regularization_lambda

        if activation_name == "cubic":
            self._activation = CubicActivation()
        elif activation_name == "sigmoid":
            self._activation = tf.keras.activations.sigmoid
        elif activation_name == "tanh":
            self._activation = tf.keras.activations.tanh
        else:
            raise Exception(f"activation_name: {activation_name} is from the known list.")

        # Trainable Variables
        # TODO(Students) Start

        # TODO(Students) End
        self._regularization_lambda = regularization_lambda
        self.hidden_dim = hidden_dim
        self.num_tokens = num_tokens
        self.embedding_dim = embedding_dim
        self.num_transitions = num_transitions
        
        self.embeddings = tf.Variable(tf.random.truncated_normal(shape=[vocab_size,embedding_dim],stddev=0.1))
        self.ip_weights = tf.Variable(tf.random.truncated_normal(shape=[self.num_tokens*self.embedding_dim,self.hidden_dim],stddev=0.1))
        self.bias = tf.Variable(tf.zeros([1,self.hidden_dim]))
        self.op_weights = tf.Variable(tf.random.truncated_normal(shape=[self.hidden_dim,self.num_transitions],stddev=1.0/math.sqrt(self.embedding_dim)))

        


    def call(self,
             inputs: tf.Tensor,
             labels: tf.Tensor = None) -> Dict[str, tf.Tensor]:
        """
        Forward pass of Dependency Parser.

        Parameters
        ----------
        inputs : ``tf.Tensor``
            Tensorized version of the batched input text. It is of shape:
            (batch_size, num_tokens) and entries are indices of tokens
            in to the vocabulary. These tokens can be word or pos tag.
            Each row corresponds to input features a configuration.
        labels : ``tf.Tensor``
            Tensor of shape (batch_size, num_transitions)
            Each row corresponds to the correct transition that
            should be made in the given configuration.

        Returns
        -------
        An output dictionary consisting of:
        logits : ``tf.Tensor``
            A tensor of shape ``(batch_size, num_transitions)`` representing
            logits (unnormalized scores) for the labels for every instance in batch.
        loss : ``tf.float32``
            If input has ``labels``, then mean loss for the batch should
            be computed and set to ``loss`` key in the output dictionary.
        """
        # TODO(Students) Start

        # TODO(Students) End
        #ip_weights = tf.Variable(tf.random.truncated_normal(shape=[self.num_tokens*self.embedding_dim,self.hidden_dim],stddev=0.1))
        embedding = tf.nn.embedding_lookup(self.embeddings,inputs)
        embedding = tf.reshape(embedding,shape=[embedding.shape[0],embedding.shape[1]*embedding.shape[2]])

        #op_weights = tf.Variable(tf.random.truncated_normal(shape=[self.hidden_dim,self.num_transitions],stddev=0.1))
        term1 = tf.matmul(embedding,self.ip_weights)
        #bias = tf.Variable(tf.zeros([1,self.hidden_dim]))
        term1 = tf.add(term1,self.bias)
        term2 = self._activation(term1)
        logits = tf.matmul(term2,self.op_weights)

        output_dict = {"logits": logits}

        if labels is not None:
            output_dict["loss"] = self.compute_loss(logits, labels)
        return output_dict

    def compute_loss(self, logits: tf.Tensor, labels: tf.Tensor) -> tf.float32:
        """
        Parameters
        ----------
        logits : ``tf.Tensor``
            A tensor of shape ``(batch_size, num_transitions)`` representing
            logits (unnormalized scores) for the labels for every instance in batch.

        Returns
        -------
        loss : ``tf.float32``
            If input has ``labels``, then mean loss for the batch should
            be computed and set to ``loss`` key in the output dictionary.

        """
        # TODO(Students) Start

        # TODO(Students) End
        #cross_entropy_loss = 
        #cross_entropy_loss = tf.negative(tf.reduce_sum(tf.math.log(logits)))
        #train_labels = tf.nn.relu(labels)
        #cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=train_labels)
        #loss = tf.reduce_mean(cross_entropy_loss) 


        #calculating L-2 Loss 
        #We know that L-2 param is theta -- so we have to include embeddings,input,
        #l2_regularisation = self._regularization_lambda*tf.nn.l2_loss(logits)
        #regularization = tf.reduce_mean(l2_regularisation)

        mask = tf.ones(shape=labels.shape)
        mask = tf.cast(mask,tf.float32)


        labels1 = (labels>-1) 
        required_term  = labels1* mask
        res = tf.nn.softmax(logits*required_term)

        mask2 = labels==1
        mask2 = tf.cast(mask2,tf.float32)

        factor = labels * mask2
        result2 = res*factor
        #loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=tf.argmax(self.labels, axis=1)))

        loss = tf.reduce_mean(tf.math.log(tf.reduce_sum(result2,axis=1)))*-1
        #loss = tf.math.log(tf.reduce_sum(result2,axis=1))
        #loss = tf.reduce_mean(loss)

        #loss = tf.negative(loss)


        
        #According to the paper, regularisation params is Theta, which is 
        #Embeddings, input_weights, op_weights, and Bias as these params are the one which get 
        #updated.
        l1 = tf.nn.l2_loss(self.embeddings)
        l2= tf.nn.l2_loss(self.ip_weights)
        l3 = tf.nn.l2_loss(self.op_weights)
        l4 = tf.nn.l2_loss(self.bias)
       
        regularization = self._regularization_lambda*(l1+l2+l3+l4)

        
        return loss + regularization
