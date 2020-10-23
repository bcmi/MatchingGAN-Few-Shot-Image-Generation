import tensorflow as tf
from dagan_architectures_with_matchingclassifier import UResNetGenerator, Discriminator
import numpy as np
import time


def Hinge_loss(real_logits, fake_logits):
    D_loss = -tf.reduce_mean(tf.minimum(0., -1.0 + real_logits)) - tf.reduce_mean(tf.minimum(0., -1.0 - fake_logits))
    G_loss = -tf.reduce_mean(fake_logits)
    return D_loss, G_loss


class DAGAN:
    def __init__(self, input_x_i, input_x_j, input_y_i, input_y_j, input_global_y_i, input_global_y_j,
                 input_x_j_selected, input_global_y_j_selected, classes, dropout_rate, generator_layer_sizes,
                 discriminator_layer_sizes, generator_layer_padding, z_inputs, z_inputs_2, matching, fce,
                 full_context_unroll_k, average_per_class_embeddings, batch_size=100, z_dim=100,
                 num_channels=1, is_training=True, augment=True, discr_inner_conv=0, gen_inner_conv=0, num_gpus=1,
                 is_z2=True, is_z2_vae=True,
                 use_wide_connections=False, selected_classes=5, support_num=5, loss_G=1, loss_D=1, loss_KL=0.0001,
                 loss_recons_B=0.01, loss_matching_G=0.01, loss_matching_D=0.01, loss_CLA=1, loss_FSL=1, loss_sim=0.01,
                 z1z2_training=True):

        """
        Initializes a DAGAN object.
        :param input_x_i: Input image x_i
        :param input_x_j: Input image x_j
        :param dropout_rate: A dropout rate placeholder or a scalar to use throughout the network
        :param generator_layer_sizes: A list with the number of feature maps per layer (generator) e.g. [64, 64, 64, 64]
        :param discriminator_layer_sizes: A list with the number of feature maps per layer (discriminator)
                                                                                                   e.g. [64, 64, 64, 64]
        :param generator_layer_padding: A list with the type of padding per layer (e.g. ["SAME", "SAME", "SAME","SAME"]
        :param z_inputs: A placeholder for the random noise injection vector z (usually gaussian or uniform distribut.)
        :param batch_size: An integer indicating the batch size for the experiment.
        :param z_dim: An integer indicating the dimensionality of the random noise vector (usually 100-dim).
        :param num_channels: Number of image channels
        :param is_training: A boolean placeholder for the training/not training flag
        :param augment: A boolean placeholder that determines whether to augment the data using rotations
        :param discr_inner_conv: Number of inner layers per multi layer in the discriminator
        :param gen_inner_conv: Number of inner layers per multi layer in the generator
        :param num_gpus: Number of GPUs to use for training
        """
        self.training = True
        self.print = False
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.z_inputs = z_inputs
        self.z_inputs_2 = z_inputs_2
        self.num_gpus = num_gpus
        self.support_num = support_num
        self.loss_G = loss_G
        self.loss_D = loss_D
        self.loss_KL = loss_KL
        self.loss_CLA = loss_CLA
        self.loss_FSL = loss_FSL
        self.loss_matching_G = loss_matching_G
        self.loss_recons_B = loss_recons_B
        self.loss_matching_D = loss_matching_D
        self.loss_sim = loss_sim
        self.input_x_i = input_x_i
        self.input_x_j = input_x_j
        self.input_x_j_selected = input_x_j_selected
        self.input_y_i = input_y_i
        self.input_y_j = input_y_j
        self.input_global_y_i = input_global_y_i
        self.input_global_y_j = input_global_y_j
        self.input_global_y_j_selected = input_global_y_j_selected
        self.classes = classes
        self.selected_classes = selected_classes
        self.dropout_rate = dropout_rate
        self.training_phase = is_training
        self.augment = augment
        self.is_z2 = is_z2
        self.is_z2_vae = is_z2_vae
        self.z1z2_training = z1z2_training

        self.g = UResNetGenerator(batch_size=self.batch_size, layer_sizes=generator_layer_sizes,
                                  num_channels=num_channels, layer_padding=generator_layer_padding,
                                  inner_layers=gen_inner_conv, name="generator", matching=matching, fce=fce,
                                  full_context_unroll_k=full_context_unroll_k,
                                  average_per_class_embeddings=average_per_class_embeddings)

        self.d = Discriminator(batch_size=self.batch_size, layer_sizes=discriminator_layer_sizes,
                               inner_layers=discr_inner_conv, use_wide_connections=use_wide_connections,
                               name="discriminator")

    def rotate_data(self, image_a, image_b):
        """
        Rotate 2 images by the same number of degrees
        :param image_a: An image a to rotate k degrees
        :param image_b: An image b to rotate k degrees
        :return: Two images rotated by the same amount of degrees
        """
        random_variable = tf.unstack(tf.random_uniform([1], minval=0, maxval=4, dtype=tf.int32, seed=None, name=None))
        image_a = tf.image.rot90(image_a, k=random_variable[0])
        image_b = tf.image.rot90(image_b, k=random_variable[0])
        return [image_a, image_b]

    def rotate_batch(self, batch_images_a, batch_images_b):
        """
        Rotate two batches such that every element from set a with the same index as an element from set b are rotated
        by an equal amount of degrees
        :param batch_images_a: A batch of images to be rotated
        :param batch_images_b: A batch of images to be rotated
        :return: A batch of images that are rotated by an element-wise equal amount of k degrees
        """
        shapes = map(int, list(batch_images_a.get_shape()))
        batch_size, x, y, c = shapes
        with tf.name_scope('augment'):
            batch_images_unpacked_a = tf.unstack(batch_images_a)
            batch_images_unpacked_b = tf.unstack(batch_images_b)
            new_images_a = []
            new_images_b = []
            for image_a, image_b in zip(batch_images_unpacked_a, batch_images_unpacked_b):
                rotate_a, rotate_b = self.augment_rotate(image_a, image_b)
                new_images_a.append(rotate_a)
                new_images_b.append(rotate_b)

            new_images_a = tf.stack(new_images_a)
            new_images_a = tf.reshape(new_images_a, (batch_size, x, y, c))
            new_images_b = tf.stack(new_images_b)
            new_images_b = tf.reshape(new_images_b, (batch_size, x, y, c))
            return [new_images_a, new_images_b]

    def generate(self, conditional_images, support_input, input_global_x_j_selected, input_y_i, input_y_j,
                 input_global_y_i, input_global_y_j_selected, selected_classes, support_num, classes, is_z2, is_z2_vae,
                 z_input=None, z_input_2=None):
        """
        Generate samples with the DAGAN
        :param conditional_images: Images to condition DAGAN on.
        :param z_input: Random noise to condition the DAGAN on. If none is used then the method will generate random
        noise with dimensionality [batch_size, z_dim]
        :return: A batch of generated images, one per conditional image
        """
        if z_input is None:
            z_input = tf.random_normal([self.batch_size, self.z_dim], mean=0, stddev=1)
            z_input_2 = tf.random_normal([self.batch_size, self.z_dim], mean=0, stddev=1)
        if self.training:
            generated_samples, z1, matching_feature, similarities, similarities_data, loss_recg, KL_loss, reconstruction_loss, crossentropy_loss_real, crossentropy_loss_fake, accuracy_real, accuracy_fake, preds_fake = self.g(
                z_input, z_input_2,
                conditional_images, support_input, input_y_i, input_y_j, selected_classes, support_num, is_z2,
                is_z2_vae,
                training=self.training_phase,
                dropout_rate=self.dropout_rate,
                z1z2_training=self.z1z2_training,
                z_dim=self.z_dim)

            return generated_samples, z1, matching_feature, similarities, similarities_data, z_input, z_input_2, loss_recg, KL_loss, reconstruction_loss, crossentropy_loss_real, crossentropy_loss_fake, accuracy_real, accuracy_fake, preds_fake
        else:
            generated_samples, similarities, similarities_data, crossentropy_loss_real, crossentropy_loss_fake, accuracy_real, accuracy_fake, preds_fake = self.g(
                z_input, z_input_2,
                conditional_images, support_input, input_y_i, input_y_j, selected_classes, support_num, is_z2,
                is_z2_vae,
                training=self.training_phase,
                dropout_rate=self.dropout_rate,
                z1z2_training=self.z1z2_training,
                z_dim=self.z_dim)

            similarities_onehot = tf.cast((0) * tf.ones_like(similarities[:, 0]), dtype=tf.int32)
            similarities_onehot = tf.expand_dims(similarities_onehot, axis=-1)
            similarities_index = tf.expand_dims(similarities[:, 0], axis=-1)

            g_same_class_outputs = self.d(generated_samples, similarities_onehot, similarities_index, input_global_x_j_selected, input_global_y_i,
                                          input_global_y_j_selected, selected_classes, support_num, classes,
                                          similarities, training=self.training_phase,
                                          dropout_rate=self.dropout_rate)
            return generated_samples, similarities,  g_same_class_outputs, preds_fake

    def augment_rotate(self, image_a, image_b):
        r = tf.unstack(tf.random_uniform([1], minval=0, maxval=2, dtype=tf.int32, seed=None, name=None))
        rotate_boolean = tf.equal(0, r, name="check-rotate-boolean")
        [image_a, image_b] = tf.cond(rotate_boolean[0], lambda: self.rotate_data(image_a, image_b),
                                     lambda: [image_a, image_b])
        return image_a, image_b

    def data_augment_batch(self, batch_images_a, batch_images_b):
        """
        Apply data augmentation to a set of image batches if self.augment is set to true
        :param batch_images_a: A batch of images to augment
        :param batch_images_b: A batch of images to augment
        :return: A list of two augmented image batches
        """

        [images_a, images_b] = tf.cond(self.augment, lambda: self.rotate_batch(batch_images_a, batch_images_b),
                                       lambda: [batch_images_a, batch_images_b])

        return images_a, images_b

    def save_features(self, name, features):
        """
        Save feature activations from a network
        :param name: A name for the summary of the features
        :param features: The features to save
        """
        for i in range(len(features)):
            shape_in = features[i].get_shape().as_list()
            channels = shape_in[3]
            y_channels = 8
            x_channels = channels / y_channels

            activations_features = tf.reshape(features[i], shape=(shape_in[0], shape_in[1], shape_in[2],
                                                                  y_channels, x_channels))

            activations_features = tf.unstack(activations_features, axis=4)
            activations_features = tf.concat(activations_features, axis=2)
            activations_features = tf.unstack(activations_features, axis=3)
            activations_features = tf.concat(activations_features, axis=1)
            activations_features = tf.expand_dims(activations_features, axis=3)
            # tf.summary.image('{}_{}'.format(name, i), activations_features)

    def loss(self, gpu_id):

        """
        Builds models, calculates losses, saves tensorboard information.
        :param gpu_id: The GPU ID to calculate losses for.
        :return: Returns the generator and discriminator losses.
        """
        #### general matching procedure
        with tf.name_scope("losses_{}".format(gpu_id)):
            before_loss = time.time()
            epsilon = 1e-8
            input_a, input_b, input_y_a, input_y_b, input_global_y_a, input_global_y_b, input_b_selected, input_global_y_b_selected = \
                self.input_x_i[gpu_id], self.input_x_j[gpu_id], self.input_y_i[gpu_id], self.input_y_j[gpu_id], \
                self.input_global_y_i[gpu_id], self.input_global_y_j[gpu_id], self.input_x_j_selected[gpu_id], \
                self.input_global_y_j_selected[gpu_id]

            # input_a_expand = tf.expand_dims(input_a,1)
            # input_a_copy = tf.tile(input_a_expand,[1,self.support_num,1,1,1])
            # current_support = tf.cond(self.z1z2_training,lambda:input_a_copy,lambda:input_b)
            current_support = input_b

            x_g, z1, matching_feature, similarities, similarities_data, z_input, z_input_2, recg_loss, KL_loss, \
            reconstruction_loss, crossentropy_loss_real, crossentropy_loss_fake, accuracy_real, accuracy_fake, preds_fake = \
                self.generate(input_a, current_support, input_b_selected, input_y_a, input_y_b, input_global_y_a,
                              input_global_y_b_selected, self.selected_classes, self.support_num, self.classes,
                              self.is_z2, self.is_z2_vae)

            similarities_minimun = tf.reduce_min(similarities, axis=1)
            one = tf.ones_like(similarities_minimun)
            zero = tf.zeros_like(similarities_minimun)
            similarities_mask = tf.where(similarities_minimun < 0, x=zero, y=one)
            labels = tf.ones_like(similarities)
            similarities_loss = -tf.reduce_mean(similarities_minimun)

            feature_total = []
            similarities_onehot = tf.cast((0) * tf.ones_like(similarities[:, 0]), dtype=tf.int32)
            similarities_onehot = tf.expand_dims(similarities_onehot, axis=-1)
            similarities_index = tf.expand_dims(similarities[:, 0], axis=-1)

            g_feature, g_same_class_outputs, g_classification_loss, g_classification_accuracy, z_feature_false = self.d(
                x_g, similarities_onehot, similarities_index,
                input_global_y_a, input_global_y_b_selected,
                self.selected_classes, self.support_num,
                self.classes, similarities, z1,
                training=self.training_phase,
                dropout_rate=self.dropout_rate)

            t_feature_total = []
            t_same_class_outputs_total = []
            t_classification_loss_total = []
            for s in range(self.support_num*self.selected_classes):
                similarities_onehot = tf.cast((s) * tf.ones_like(similarities[:, s]), dtype=tf.int32)
                similarities_onehot = tf.expand_dims(similarities_onehot, axis=-1)
                similarities_index = tf.expand_dims(similarities[:, s], axis=-1)
                t_feature, t_same_class_outputs, t_classification_loss, t_classification_accuracy, z_feature_true = self.d(
                    input_b[:, s], similarities_onehot, similarities_index,
                    input_global_y_b[:,s], input_global_y_b_selected,
                    self.selected_classes, self.support_num,
                    self.classes, similarities, z1,
                    training=self.training_phase,
                    dropout_rate=self.dropout_rate)

                feature_total.append(t_feature)
                t_same_class_outputs_total.append(t_same_class_outputs)
                t_classification_loss_total.append(t_classification_loss)

            # feature_total = tf.stack(feature_total, axis=1)
            # feature_total = self.g.classify(similarities, support_set_y=feature_total, name='classify')
            # feature_loss = tf.reduce_mean(tf.square(feature_total - g_feature))
            feature_total = tf.stack(feature_total, axis=1)
            feature_total_reshape = tf.reshape(feature_total,
                                               [feature_total.get_shape()[0], feature_total.get_shape()[1],
                                                feature_total.get_shape()[2] * \
                                                feature_total.get_shape()[3] * feature_total.get_shape()[4]])
            feature_total_reshape = self.g.classify(similarities, support_set_y=feature_total_reshape, name='classify')
            feature_total = tf.reshape(feature_total_reshape,
                                       [feature_total.get_shape()[0], feature_total.get_shape()[2],
                                        feature_total.get_shape()[3], feature_total.get_shape()[4]])
            feature_loss = tf.reduce_mean(tf.abs(feature_total - g_feature))
            

            

            t_same_class_outputs = tf.concat(t_same_class_outputs_total, axis=0)
            # t_same_class_outputs = tf.stack(t_same_class_outputs_total,axis=1)
            # t_same_class_outputs = self.g.classify(similarities, support_set_y=t_same_class_outputs, name='classify')

            



            t_classification_loss = tf.concat(t_classification_loss_total, axis=0)

            d_fake = g_same_class_outputs
            d_real = t_same_class_outputs
            d_loss_pure, G_loss = Hinge_loss(d_real, d_fake)

            if self.print:
                # similarities = tf.Print(similarities,[similarities],'similarities')
                # similarities_l1_loss = tf.Print(similarities_l1_loss,[similarities_l1_loss],'similarities_l1_loss')
                # g_same_class_outputs = tf.Print(g_classification_loss,[g_same_class_outputs],'the probability of the generated image')
                # t_same_class_outputs = tf.Print(t_same_class_outputs,[t_same_class_outputs],'the probability of the real image')
                #### generator
                # similarities_mask = tf.Print(similarities_mask,[similarities_mask],'similarity mask')
                g_classification_loss = tf.Print(g_classification_loss, [g_classification_loss], 'classification_loss',
                                                 summarize=5)
                reconstruction_loss = tf.Print(reconstruction_loss, [reconstruction_loss], 'l1_reconstruction_loss',
                                               summarize=5)
                feature_loss = tf.Print(feature_loss, [feature_loss], 'matching_D_loss', summarize=5)
                d_loss_pure = tf.Print(d_loss_pure, [d_loss_pure], 'D_loss', summarize=5)
                G_loss = tf.Print(G_loss, [G_loss], 'G_loss', summarize=5)

            ##### without mask
            loss_KL = tf.reduce_mean(KL_loss)
            loss_recg = tf.reduce_mean(recg_loss)
            loss_reconstruction = tf.reduce_mean(reconstruction_loss)
            # loss_reconstruction_randomz1 = tf.reduce_mean(reconstruction_loss_randomz1)
            # crossentropy_loss_fake = tf.reduce_mean(crossentropy_loss_fake)
            loss_feature = tf.reduce_mean(feature_loss)
            g_classification_loss = tf.reduce_mean(g_classification_loss)
            t_classification_loss = tf.reduce_mean(t_classification_loss)

            g_loss_z1z2_training = self.loss_recons_B * loss_reconstruction

            g_loss_z1z2_notraining = G_loss * self.loss_G + g_classification_loss * self.loss_CLA + self.loss_recons_B * loss_reconstruction + \
                                     self.loss_matching_D * loss_feature

            # g_loss = tf.cond(self.z1z2_training, lambda: g_loss_z1z2_training,
            #                  lambda: g_loss_z1z2_notraining)
            g_loss = g_loss_z1z2_notraining

            # recons_loss = self.loss_recons_B*loss_reconstruction + self.loss_KL*loss_KL

            # g_loss_z1z2_notraining = -tf.reduce_mean(d_fake*similarities_mask)*self.loss_G - g_classification_loss*self.loss_CLA - \
            #   self.loss_matching_G* loss_recg - self.loss_recons_B*loss_reconstruction_randomz1 - self.loss_FSL * crossentropy_loss_fake - self.loss_matching_D * loss_feature

            # g_loss_z1z2_notraining = -tf.reduce_mean(d_fake)*self.loss_G - g_classification_loss*self.loss_CLA - \
            # self.loss_recons_S * loss_recg

            # g_loss = g_loss_z1z2_notraining

            # alpha = tf.random_uniform(
            #     shape=[self.batch_size, 1],
            #     minval=0.,
            #     maxval=1.
            # )

            # input_shape = input_a.get_shape()
            # input_shape = [int(n) for n in input_shape]
            # differences_g = x_g - input_a
            # differences_g = tf.reshape(differences_g, (self.batch_size, input_shape[1]*input_shape[2]*input_shape[3]))
            # interpolates_g = input_a + tf.reshape(alpha * differences_g, (self.batch_size, input_shape[1],
            #                                                               input_shape[2], input_shape[3]))
            # interpolated_feature, pre_grads, pre_classification_loss, pre_classification_acc,_ = self.d(interpolates_g, input_b_selected,input_global_y_a,input_global_y_b_selected,self.selected_classes,self.support_num, self.classes, similarities, z1, dropout_rate=self.dropout_rate,
            #                      training=self.training_phase)

            # # interpolated_feature, pre_grads = self.d(interpolates_g, input_b, input_global_y_a,input_global_y_b_selected,self.selected_classes,self.support_num, self.classes, similarities, dropout_rate=self.dropout_rate,
            # #                      training=self.training_phase)

            # gradients = tf.gradients(tf.reduce_mean(pre_grads,axis=[1,2,3]), interpolates_g)[0]
            # slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
            # gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
            # if self.print:
            #     gradient_penalty = tf.Print(gradient_penalty,[gradient_penalty],'gradient_penalty')

            d_loss = self.loss_D * (d_loss_pure) + self.loss_CLA * t_classification_loss

            # tf.add_to_collection('fzl_losses',crossentropy_loss_real)

            tf.add_to_collection('g_losses', g_loss)
            tf.add_to_collection('d_losses', d_loss)
            tf.add_to_collection('c_losses', t_classification_loss)
            # tf.add_to_collection('recons_loss', recons_loss)

            tf.summary.scalar('G_losses', G_loss)
            tf.summary.scalar('D_losses', d_loss_pure)
            tf.summary.scalar('total_g_losses', g_loss)
            tf.summary.scalar('total_d_losses', d_loss)
            tf.summary.scalar('c_losses', g_classification_loss)
            tf.summary.scalar('reconstruction_losses', loss_reconstruction)
            tf.summary.scalar('matchingD_losses', loss_feature)

        return {
            "g_losses": tf.add_n(tf.get_collection('g_losses'), name='total_g_loss'),
            "d_losses": tf.add_n(tf.get_collection('d_losses'), name='total_d_loss'),
            "c_losses": tf.add_n(tf.get_collection('c_losses'), name='total_c_loss'),
            # "fzl_losses":tf.add_n(tf.get_collection('fzl_losses'),name='total_fzl_loss'),
            # "recons_losses":tf.add_n(tf.get_collection('recons_losses'),name='total_recons_loss'),
        }

    def train(self, opts, losses):

        """
        Returns ops for training our DAGAN system.
        :param opts: A dict with optimizers.
        :param losses: A dict with losses.
        :return: A dict with training ops for the dicriminator and the generator.
        """
        opt_ops = dict()
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            opt_ops["g_opt_op"] = opts["g_opt"].minimize(losses["g_losses"],
                                                         var_list=self.g.variables,
                                                         colocate_gradients_with_ops=True)
            opt_ops["d_opt_op"] = opts["d_opt"].minimize(losses["d_losses"],
                                                         var_list=self.d.variables_d,
                                                         colocate_gradients_with_ops=True)
            opt_ops["c_opt_op"] = opts["c_opt"].minimize(losses['c_losses'], var_list=self.d.variables_c,
                                                         colocate_gradients_with_ops=True)

            # opt_ops["fzl_opt_op"] = opts["fzl_opt"].minimize(losses['fzl_losses'], var_list=self.g.variables_fzl,
            #                                              colocate_gradients_with_ops=True)
            # opt_ops["recons_opt_op"] = opts["recons_opt"].minimize(losses['recons_losses'], var_list=self.g.variables,
            #                                              colocate_gradients_with_ops=True)

        return opt_ops

    def init_train(self, learning_rate=1e-4, beta1=0.0, beta2=0.9):
        """
        Initialize training by constructing the summary, loss and ops
        :param learning_rate: The learning rate for the Adam optimizer
        :param beta1: Beta1 for the Adam optimizer
        :param beta2: Beta2 for the Adam optimizer
        :return: summary op, losses and training ops.
        """

        losses = dict()
        opts = dict()

        if self.num_gpus > 0:
            device_ids = ['/gpu:{}'.format(i) for i in range(self.num_gpus)]
        else:
            device_ids = ['/cpu:0']
        for gpu_id, device_id in enumerate(device_ids):
            with tf.device(device_id):
                total_losses = self.loss(gpu_id=gpu_id)
                for key, value in total_losses.items():
                    if key not in losses.keys():
                        losses[key] = [value]
                    else:
                        losses[key].append(value)

        for key in list(losses.keys()):
            losses[key] = tf.reduce_mean(losses[key], axis=0)
            opts[key.replace("losses", "opt")] = tf.train.AdamOptimizer(beta1=beta1, beta2=beta2,
                                                                        learning_rate=learning_rate)

            # opts[key.replace("losses", "opt")] = tf.train.RMSPropOptimizer(learning_rate=learning_rate)

        summary = tf.summary.merge_all()
        apply_grads_ops = self.train(opts=opts, losses=losses)

        return summary, losses, apply_grads_ops

    def sample_same_images(self):
        """
        Samples images from the DAGAN using input_x_i as image




        conditional input and z_inputs as the gaussian noise.
        :return: Inputs and generated images
        """
        conditional_inputs = self.input_x_i[0]
        support_input = self.input_x_j[0]
        input_global_y_i = self.input_global_y_i[0]
        input_global_x_j_selected = self.input_x_j_selected[0]

        input_y_i = self.input_y_i[0]
        input_y_j = self.input_y_j[0]
        input_global_y_j = self.input_global_y_j[0]
        input_global_y_j_selected = self.input_global_y_j_selected[0]

        classes = self.classes
        #### calculating the d_loss for score of selected samples

        if self.training:
            generated, f_encode_z, matching_feature, similarities, similarities_data, z_input, z_input_2, loss_recg, KL_loss, reconstruction_loss, crossentropy_loss_real, crossentropy_loss_fake, accuracy_real, accuracy_fake, preds_fake = \
                self.generate(conditional_images=conditional_inputs,
                              support_input=support_input,
                              input_global_y_i=input_global_y_i,
                              input_global_x_j_selected=input_global_x_j_selected,
                              input_y_i=input_y_i,
                              input_y_j=input_y_j,
                              input_global_y_j_selected=input_global_y_j_selected,
                              selected_classes=self.selected_classes,
                              support_num=self.support_num,
                              classes=classes,
                              z_input=self.z_inputs,
                              z_input_2=self.z_inputs_2,
                              is_z2=self.is_z2,
                              is_z2_vae=self.is_z2_vae)
            return self.input_x_i[0], self.input_x_j[
                0], generated, generated, generated,  input_y_i, input_global_y_i
        else:
            generated, similarities, d_loss, preds_fake = self.generate(
                conditional_images=conditional_inputs,
                support_input=support_input,
                input_global_y_i=input_global_y_i,
                input_global_x_j_selected=input_global_x_j_selected,
                input_y_i=input_y_i,
                input_y_j=input_y_j,
                input_global_y_j_selected=input_global_y_j_selected,
                selected_classes=self.selected_classes,
                support_num=self.support_num,
                classes=classes,
                z_input=self.z_inputs,
                z_input_2=self.z_inputs_2,
                is_z2=self.is_z2,
                is_z2_vae=self.is_z2_vae)


            return self.input_x_i[0], self.input_x_j[
                0], generated, input_y_i, input_global_y_i, similarities, similarities, similarities