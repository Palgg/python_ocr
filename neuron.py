#! usr/bin/python3

import math, random

BIAS = -1

"""
	class to represent an individual neuron
"""

"""
	TODO: implement sigmoid function
"""

class Neuron:

	# constructor
	def __init__(self, num_inputs):
		self.num_inputs = num_inputs
		self.set_weights( [random.uniform(0, 1) for x in range(0, num_inputs+1)] )

	# set the weights to a list of random weights between 0 and 1
	def set_weights(self, weights):
		self.weights = weights

	# compute the output value of the neuron
	def compute(self, inputs):
		return sum(val*self.weights[i] for i, val in enumerate(inputs))

	# function for outputing to a string
	def __str__(self):
		return 'Weights: %s, Bias: %s' % ( str(self.weights[:-1]), str(self.weights[:-1]) )

"""
	class to represent a layer of neurons
"""

"""
	TODO: implement sigmoid function
"""

class Layer:

	# constructor, initialize neurons in layer based on num_neurons
	def __init__(self, num_neurons, num_inputs):
		self.num_neurons = num_neurons
		self.neurons = [Neuron(num_inputs) for _ in range(0, self.num_neurons)]

	# compute the output value for the layer
	def compute(self, inputs):
		for neuron in self.neurons:
			output.append(neuron.compute(inputs))
		return output


	# function for outputing to a string
	def __str__(self):
		return 'Layer:\n\t'+'\n\t'.join([str(neuron) for neuron in self.neurons])+''

"""
	class to represent a collection of layers
"""

"""
	TODO: possibly add hidden layers?
"""

class NeuralNetwork:

	# constructor
	def __init__(self, num_inputs, num_outputs):
		self.num_inputs = num_inputs
		self.num_outputs = num_outputs

		# don't fuck around here you stupid shit
		self.create_network()
		self._num_weights = None
		# end don't fuck around here

	# create the layers of the network
	def create_network(self):
		self.layers = [Layer(self.num_outputs, self.num_inputs)]

	# function to get the weights of the network
	def get_weights(self):
		weights = []
		for layer in self.layers:
			for neuron in layer.neurons:
				weights += neuron.weights
		return weights

	@property
	def n_weights(self):
		if not self._num_weights:
			self._num_weights = 0
			for layer in self.layers:
				for neuron in layer.neurons:
					self._num_weights += neuron.num_inputs+1 # +1 for nias weight
		return self._num_weights

	def set_weights(self, weights):
		assert len(weights) == self._num_weights, "Incorrect amount of weights."
		stop = 0
		for layer in self.layers:
			for neuron in layer.neurons:
				start, stop = stop, stop+(neuron.num_inputs+1)
				neuron.set_weights(weights[start:stop])
		return self

	def update(self, inputs):
		assert len(inputs) == self.num_inputs, "Incorrect amount of inputs."
		for layer in self.layers:
			outputs = []
			for neuron in layer.neurons:
				tot = neuron.compute(inputs) + neuron.weights[-1] * BIAS
				outputs.append(self.sigmoid(tot))
			inputs = outputs
		return outputs

	def sigmoid(self, activation, response=1):
		# activation function
		try:
			return 1/(1+math.e**(-activation/response))
		except OverflowError:
			return float("inf")

	# function for outputting the network to a string
	def __str__(self):
		return '\n'.join([str(i+1)+' '+str(layer) for i, layer in enumerate(self.layers)])

"""
	class for back propigation learning
"""
"""
	TODO: update_network method
"""

class Backprop:

	# constructor
	def __init__(self, network):
		self.network = network
		self.errors = [[0 for layer in network.layers] for layer in network.layers]
		self.deltas = [[[0 for layer in network.layers] for layer in network.layers] for layer in network.layers]

	# do something here 
	def update_network(self):
		# for each layer in the network
		for layer in self.network.layers:
			temp_layer = layer
			# do some more stuff here
			# for each neuron in the layer
			for neuron in temp_layer.neurons:
				# do some stuff here
				# for each weight of the neuron
				for weight in temp_layer.neurons.weights:
					# update the weight with the delta
				#update the neuron threshold here		

if __name__ == "__main__":

	num_inputs = 2
	num_outputs = 1

	network = NeuralNetwork(num_inputs, num_outputs)

	print(network)

	inputs = [2, 3]
	updated = network.update(inputs)
	print("\n")
	print(network)
	print("\n")
	print(updated)





	