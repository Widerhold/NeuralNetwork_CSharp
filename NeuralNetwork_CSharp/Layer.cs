using System;

namespace NeuralNetwork_CSharp
{
    public class Layer
    {
        private static readonly Random Random = new Random();
        private const double LEARNING_RATE = 0.1f;

        private readonly int _numberOfInputs;
        private readonly int _numberOfOutputs;
        private readonly Func<double, double> _activationFunction;
        private readonly double[] _outputs;
        private double[] _inputs;
        public readonly double[,] Weights;

        /// <summary>
        /// Creates a Layer with the amount of provided inputs and outputs.
        /// </summary>
        /// <param name="numberOfInputs"></param>
        /// <param name="numberOfOutputs"></param>
        /// <param name="activationFunction"></param>
        public Layer(int numberOfInputs, int numberOfOutputs, Func<double, double> activationFunction)
        {
            _numberOfInputs = numberOfInputs;
            _numberOfOutputs = numberOfOutputs;
            _outputs = new double[numberOfOutputs];
            _inputs = new double[numberOfInputs];
            Weights = CreateWeights(numberOfOutputs, numberOfInputs);
            _activationFunction = activationFunction;
        }

        /// <summary>
        /// Creates a Weighted Matrix of equal of the Size of Outputs * Inputs
        /// </summary>
        /// <param name="numberOfOuputs">Amount of Outputs</param>
        /// <param name="numberOfInputs">Amount of Inputs</param>
        /// <returns></returns>
        private static double[,] CreateWeights(int numberOfOuputs, int numberOfInputs)
        {
            double[,] weights = new double[numberOfOuputs, numberOfInputs];
            for (int i = 0; i < numberOfOuputs; i++)
                for (int j = 0; j < numberOfInputs; j++)
                    weights[i, j] = Random.NextDouble() - 0.5f;
            return weights;
        }

        /// <summary>
        /// Calculates the Output for an given input 
        /// </summary>
        /// <param name="inputs"></param>
        /// <returns></returns>
        public double[] Forward(double[] inputs)
        {
            _inputs = inputs;
            for (int i = 0; i < _numberOfOutputs; i++)
            {
                _outputs[i] = 0;
                for (int j = 0; j < _numberOfInputs; j++)
                    _outputs[i] += _inputs[j] * Weights[i, j];
                _outputs[i] = Math.Tanh(_outputs[i]);
            }
            return _outputs;
        }

        /// <summary>
        /// Calculates the gamma and updates the weights via the calculated error. Used for the Output Layer
        /// </summary>
        /// <param name="expected"></param>
        /// <returns></returns>
        public double[] BackPropagateOutput(double[] expected)
        {
            double[] gamma = new double[_numberOfOutputs];
            for (int i = 0; i < _numberOfOutputs; i++)
            {
                double error = _outputs[i] - expected[i];
                gamma[i] = error * _activationFunction(_outputs[i]);
                for (int j = 0; j < _numberOfInputs; j++)
                {
                    double weightsDelta = gamma[i] * _inputs[j];
                    Weights[i, j] = UpdateWeights(Weights[i, j], weightsDelta, LEARNING_RATE);
                }

            }
            return gamma;
        }

        /// <summary>
        /// Calculates the gamma and updates the weights via the calculated error. Used for the Hidden Layer
        /// </summary>
        /// <param name="gammaForward"></param>
        /// <param name="weightsForward"></param>
        /// <returns></returns>
        public double[] BackPropagateHidden(double[] gammaForward, double[,] weightsForward)
        {
            double[] gamma = new double[_numberOfOutputs];
            for (int i = 0; i < _numberOfOutputs; i++)
            {
                for (int j = 0; j < gammaForward.Length; j++)
                    gamma[i] += gammaForward[j] * weightsForward[j, i];
                gamma[i] *= _activationFunction(_outputs[i]);
                for (int j = 0; j < _numberOfInputs; j++)
                {
                    double weightsDelta = gamma[i] * _inputs[j];
                    Weights[i, j] = UpdateWeights(Weights[i, j], weightsDelta, LEARNING_RATE);
                }

            }
            return gamma;
        }

        /// <summary>
        /// Calculates the new weight by subtracting the supplyied delta multiplied by the rate of learning 
        /// </summary>
        /// <param name="weight"></param>
        /// <param name="weightsDelta"></param>
        /// <param name="learningRate"></param>
        /// <returns></returns>
        private double UpdateWeights(double weight, double weightsDelta, double learningRate)
        {
            return weight - weightsDelta * learningRate;
        }
    }
}
