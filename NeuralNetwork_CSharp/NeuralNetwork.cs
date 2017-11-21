using System;

namespace NeuralNetwork_CSharp
{
    public class NeuralNetwork
    {
        private readonly Layer[] _layers;

        public NeuralNetwork(int[] layer)
        {
            _layers = new Layer[layer.Length - 1];
            for (int i = 0; i < _layers.Length; i++)
            {
                _layers[i] = new Layer(layer[i], layer[i + 1], TanhDerivation);
            }
        }

        /// <summary>
        /// Calculates the output for the given inputs by passing the values through each layer
        /// </summary>
        /// <param name="inputs"></param>
        /// <returns></returns>
        public double[] Forward(double[] inputs)
        {
            double[] output = _layers[0].Forward(inputs);
            for (int i = 1; i < _layers.Length; i++)
            {
                output = _layers[i].Forward(output);
            }
            return output;
        }

        /// <summary>
        /// Calculates the error throught Backprogpagation and adjusts the weights
        /// </summary>
        /// <param name="expected"></param>
        public void BackPropagate(double[] expected)
        {
            double[] gamma = _layers[_layers.Length - 1].BackPropagateOutput(expected);
            for (int i = _layers.Length - 2; i >= 0; i--)
                gamma = _layers[i].BackPropagateHidden(gamma, _layers[i + 1].Weights);
        }

        public static double TanhDerivation(double value)
        {
            return 1 - (value * value);
        }

        public static double Sigmoid(double value)
        {
            return 2.0f / (1.0f + Math.Exp(-2.0f * value))-1.0f;
        }
    }
}
