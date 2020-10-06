import numpy as np
import copy
from keras import layers, models, optimizers
from keras import backend as K

class Critic:
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size

        # Initialize any other variables here

        self.build_model()

    def build_model(self):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # Input layers
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')

        # Hidden layer(s) for state pathway        
        net_states = layers.Dense(units=32, activation=None)(states)
#         net_states = layers.BatchNormalization()(net_states)
        net_states = layers.Dropout(0.1)(net_states)
        net_states = layers.Activation(activation='relu')(net_states)
        
        net_states = layers.Dense(units=64, activation=None)(net_states)
#         net_states = layers.BatchNormalization()(net_states)
        net_states = layers.Dropout(0.1)(net_states)
        net_states = layers.Activation(activation='relu')(net_states)

        # Hidden layer(s) for action pathway        
        net_actions = layers.Dense(units=32, activation=None)(actions)
#         net_actions = layers.BatchNormalization()(net_actions)
        net_actions = layers.Dropout(0.1)(net_actions)
        net_actions = layers.Activation(activation='relu')(net_actions)
        
        net_actions = layers.Dense(units=64, activation=None)(net_actions)
#         net_actions = layers.BatchNormalization()(net_actions)
        net_actions = layers.Dropout(0.1)(net_actions)
        net_actions = layers.Activation(activation='relu')(net_actions)

        # State and action pathways combined
        net = layers.Add()([net_states, net_actions])
        net = layers.Activation('relu')(net)

        net = layers.Dense(units=64, activation='relu')(net)
        net = layers.Dropout(0.1)(net)

        # Final output layer to prduce action values (Q values)
        Q_values = layers.Dense(units=1, name='q_values')(net)

        # Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        # Optimizer and compiled model for training with built-in loss function
        optimizer = optimizers.Adam(lr=0.01)
        self.model.compile(optimizer=optimizer, loss='mse')

        # Action gradients (derivative of Q values w.r.t. to actions)
        action_gradients = K.gradients(Q_values, actions)

        # Function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)
        
        
######################################################################################################################################

class Actor:
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, action_low, action_high):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            action_low (array): Min value of each action dimension
            action_high (array): Max value of each action dimension
        """
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low

        # Initialize any other variables here

        self.build_model()

    def build_model(self):
        """Build an actor (policy) network that maps states -> actions."""
        # Input layer (states)
        states = layers.Input(shape=(self.state_size,), name='states')

        # Hidden layers
        net = layers.Dense(units=32, activation=None)(states)
#         net = layers.BatchNormalization()(net)
        net = layers.Dropout(0.1)(net)
        net = layers.Activation(activation='relu')(net)
        net = layers.Dense(units=64, activation=None)(net)
#         net = layers.BatchNormalization()(net)
        net = layers.Dropout(0.1)(net)
        net = layers.Activation(activation='relu')(net)
        net = layers.Dense(units=128, activation='relu')(net)
        net = layers.Dropout(0.1)(net)

        # Final output layer with sigmoid activation
        raw_actions = layers.Dense(units=self.action_size, activation='sigmoid',
            name='raw_actions')(net)

        # Scale [0, 1] output for each action dimension to proper range
        actions = layers.Lambda(lambda x: (x * self.action_range) + self.action_low,
            name='actions')(raw_actions)

        # Keras model
        self.model = models.Model(inputs=states, outputs=actions)

        # Loss function using action value (Q value) gradients
        action_gradients = layers.Input(shape=(self.action_size,))
        loss = K.mean(-action_gradients * actions)

        # Optimizer and training function
        optimizer = optimizers.Adam(lr=0.001)
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = K.function(
            inputs=[self.model.input, action_gradients, K.learning_phase()],
            outputs=[],
            updates=updates_op)
        
###############################################################################################################################
# Modelling the noise

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu, theta, sigma):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state


