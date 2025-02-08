# ATTENTION!
# ===========================================================
# The MO-GAAL class is inherited from BaseDetector class.
# Parameter details:
#   k: the number of generators
#   stop_epochs: the number of iterations to train the
#                discriminator and generators.
#   lr_d: Learning rate of discriminator
#   lr_g: Learning rate of generators
#
# Note! If you don't change these parameters, the default
# values will be used.
# -----------------------------------------------------------
class MO_GAAL(BaseDetector):

    # :::::::::::::::::::::: Constructor :::::::::::::::::::::::
    def __init__(self, k=10, stop_epochs=20, lr_d=0.01,
                 lr_g=0.0001, decay=1e-6, momentum=0.9,
                 contamination=0.1):
        super(MO_GAAL, self).__init__(contamination=contamination)
        self.k = k
        self.stop_epochs = stop_epochs
        self.lr_d = lr_d
        self.lr_g = lr_g
        self.decay = decay
        self.momentum = momentum
        self.generated_outliers = 0;


    # ::::::::::::::::: Fit model on data (X) :::::::::::::::::
    def fit(self, X, y=None):
        """   Parameters
              ----------      X : numpy array of shape (n_samples, n_features)
                                  The input samples.
                              y : Ignored (in unsupervised methods)
                                  Not used, present for API consistency by convention.
              Returns
              -------         self : object (Fitted estimator.)   """

        X = check_array(X)
        self._set_n_classes(y)
        self.train_history = defaultdict(list)
        names = locals()
        stop = 0

        # ===================== Print some information  ===================
        epochs = self.stop_epochs
        latent_size = X.shape[1]
        data_size = X.shape[0]
        print("Number of epochs = ", epochs)
        print("latent_size = ", latent_size)
        print("data_size = ", data_size)

        # ===================== Create discriminator  =====================
        self.discriminator = create_discriminator(latent_size, data_size)
        self.discriminator.compile(
            optimizer=SGD(lr=self.lr_d, decay=self.decay, momentum=self.momentum),
            loss='binary_crossentropy'
        )

        # =================== Create k combine models =====================
        for i in range(self.k):
            names['sub_generator' + str(i)] = create_generator(latent_size)
            latent = Input(shape=(latent_size,))
            names['fake' + str(i)] = names['sub_generator' + str(i)](latent)
            self.discriminator.trainable = False
            names['fake' + str(i)] = self.discriminator(names['fake' + str(i)])
            names['combine_model' + str(i)] = Model(latent,
                                                    names['fake' + str(i)])
            names['combine_model' + str(i)].compile(
                optimizer=SGD(lr=self.lr_g, decay=self.decay, momentum=self.momentum),
                loss='binary_crossentropy'
            )

        # ======================== Start iteration ========================
        for epoch in range(epochs):
            print('Epoch {} of {}'.format(epoch + 1, epochs))
            batch_size = min(500, data_size)
            num_batches = int(data_size / batch_size)

            for index in range(num_batches):
                print('\nTraining for epoch {} index {}:'.format(epoch + 1,
                                                                 index + 1))

                # ++++++++++++++++++ Generate noise +++++++++++++++++++
                noise_size = batch_size
                noise = np.random.uniform(0, 1, (int(noise_size), latent_size))

                # +++++++++++++++++ Get training data +++++++++++++++++
                data_batch = X[index * batch_size: (index + 1) * batch_size]

                # ++++++++++++ Generate potential outliers ++++++++++++
                block = ((1 + self.k) * self.k) // 2
                for i in range(self.k):  # k: The number of sub generators.
                    if i != (self.k - 1):
                        noise_start = int(
                            (((self.k + (self.k - i + 1)) * i) / 2) * (
                                    noise_size // block))
                        noise_end = int(
                            (((self.k + (self.k - i)) * (i + 1)) / 2) * (
                                    noise_size // block))
                        names['noise' + str(i)] = noise[noise_start:noise_end]
                        names['generated_data' + str(i)] = names[
                            'sub_generator' + str(i)].predict(
                            names['noise' + str(i)], verbose=0)
                    else:
                        noise_start = int(
                            (((self.k + (self.k - i + 1)) * i) / 2) * (
                                    noise_size // block))
                        names['noise' + str(i)] = noise[noise_start:noise_size]
                        names['generated_data' + str(i)] = names[
                            'sub_generator' + str(i)].predict(
                            names['noise' + str(i)], verbose=0)

                # ++++++++ Concatenate real data to generated data +++++++
                for i in range(self.k):
                    if i == 0:
                        x = np.concatenate(
                            (data_batch, names['generated_data' + str(i)]))
                    else:
                        x = np.concatenate(
                            (x, names['generated_data' + str(i)]))
                y = np.array([1] * batch_size + [0] * int(noise_size))  # 1: real_data,  0: noise_data

                # +++++++++++++++++ Train discriminator ++++++++++++++++++
                discriminator_loss = self.discriminator.train_on_batch(x, y)
                self.train_history['discriminator_loss'].append(
                    discriminator_loss)

                # +++++++++ Get the target value of sub-generator +++++++++
                pred_scores = self.discriminator.predict(X).ravel()

                for i in range(self.k):
                    names['T' + str(i)] = np.percentile(pred_scores,
                                                        i / self.k * 100)
                    names['trick' + str(i)] = np.array(
                        [float(names['T' + str(i)])] * noise_size)

                # ++++++++++++++++++++ Train generator ++++++++++++++++++++
                noise = np.random.uniform(0, 1, (int(noise_size), latent_size))
                if stop == 0:
                    for i in range(self.k):
                        names['sub_generator' + str(i) + '_loss'] = \
                            names['combine_model' + str(i)].train_on_batch(
                                noise, names['trick' + str(i)])
                        self.train_history[
                            'sub_generator{}_loss'.format(i)].append(
                            names['sub_generator' + str(i) + '_loss'])
                else:
                    for i in range(self.k):
                        names['sub_generator' + str(i) + '_loss'] = names[
                            'combine_model' + str(i)].evaluate(noise, names[
                            'trick' + str(i)])
                        self.train_history[
                            'sub_generator{}_loss'.format(i)].append(
                            names['sub_generator' + str(i) + '_loss'])

                generator_loss = 0
                for i in range(self.k):
                    generator_loss = generator_loss + names[
                        'sub_generator' + str(i) + '_loss']
                generator_loss = generator_loss / self.k
                self.train_history['generator_loss'].append(generator_loss)

                # +++++++++++++ Concatenate all generated data +++++++++++++
                # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                if epoch == self.stop_epochs - 1:
                    for i in range(self.k):
                        if (index == 0 and i == 0):
                            self.generated_outliers = names['generated_data' + str(i)]
                        else:
                            self.generated_outliers = np.concatenate(
                                (self.generated_outliers, names['generated_data' + str(i)]))

                # ++++++++++++++++ Stop training generator ++++++++++++++++
                if epoch + 1 > self.stop_epochs:
                    stop = 1

            # ============= Plot Discriminator & Generator Loss ================
            # ==================================================================
            HorizontalX = np.arange(len(self.train_history['generator_loss']))
            plt.plot(HorizontalX, self.train_history['generator_loss'],
                     'r--', label="Generator Loss")
            plt.plot(HorizontalX, self.train_history['discriminator_loss'],
                     'b', label="Discriminator_loss")
            plt.legend(loc='upper right')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')

            # save the figure
            if epoch == epochs - 1:
                plt.savefig('ِCost_GAN.png', dpi=300, bbox_inches='tight')
            plt.show()

        # ======================== Detection result ========================
        self.decision_scores_ = self.discriminator.predict(X).ravel()
        self._process_decision_scores()

        return self


    # :::::::::::::::::::::: Discriminator's outputs on data (X) :::::::::::::::::::::
    def decision_function(self, X):
        """ Parameters
            ----------    X : numpy array of shape (n_samples, n_features)
                              The training input samples. Sparse matrices are accepted only
                              if they are supported by the base estimator.
            Returns
            -------       anomaly_scores : numpy array of shape (n_samples,)
                              The anomaly score of the input samples.        """

        check_is_fitted(self, ['discriminator'])
        X = check_array(X)
        pred_scores = self.discriminator.predict(X).ravel()
        return pred_scores