using System;

namespace NeuralNetwork_CSharp
{
    public class Layer
    {
        private static readonly Random Random = new Random();
        private const float LEARNING_RATE = 0.1f;

        private readonly int _numberOfInputs;
        private readonly int _numberOfOutputs;
        private readonly Func<float, float> _activationFunction;
        private readonly float[] _outputs;
        private float[] _inputs;
        public readonly float[,] Weights;

        /// <summary>
        /// Creates a Layer with the amount of provided inputs and outputs.
        /// </summary>
        /// <param name="numberOfInputs"></param>
        /// <param name="numberOfOutputs"></param>
        /// <param name="activationFunction"></param>
        public Layer(int numberOfInputs, int numberOfOutputs, Func<float, float> activationFunction)
        {
            _numberOfInputs = numberOfInputs;
            _numberOfOutputs = numberOfOutputs;
            _outputs = new float[numberOfOutputs];
            _inputs = new float[numberOfInputs];
            Weights = CreateWeights(numberOfOutputs, numberOfInputs);
            _activationFunction = activationFunction;
        }

        /// <summary>
        /// Creates a Weighted Matrix of equal of the Size of Outputs * Inputs
        /// </summary>
        /// <param name="numberOfOuputs">Amount of Outputs</param>
        /// <param name="numberOfInputs">Amount of Inputs</param>
        /// <returns></returns>
        private static float[,] CreateWeights(int numberOfOuputs, int numberOfInputs)
        {
            float[,] weights = new float[numberOfOuputs, numberOfInputs];
            for (int i = 0; i < numberOfOuputs; i++)
                for (int j = 0; j < numberOfInputs; j++)
                    weights[i, j] = (float)Random.NextDouble() - 0.5f;
            return weights;
        }

        /// <summary>
        /// Calculates the Output for an given input 
        /// </summary>
        /// <param name="inputs"></param>
        /// <returns></returns>
        public float[] Forward(float[] inputs)
        {
            _inputs = inputs;
            for (int i = 0; i < _numberOfOutputs; i++)
            {
                _outputs[i] = 0;
                for (int j = 0; j < _numberOfInputs; j++)
                    _outputs[i] += _inputs[j] * Weights[i, j];
                _outputs[i] = (float)Math.Tanh(_outputs[i]);
            }
            return _outputs;
        }

        /// <summary>
        /// Calculates the gamma and updates the weights via the calculated error. Used for the Output Layer
        /// </summary>
        /// <param name="expected"></param>
        /// <returns></returns>
        public float[] BackPropagateOutput(float[] expected)
        {
            float[] gamma = new float[_numberOfOutputs];
            for (int i = 0; i < _numberOfOutputs; i++)
            {
                float error = _outputs[i] - expected[i];
                gamma[i] = error * _activationFunction(_outputs[i]);
                for (int j = 0; j < _numberOfInputs; j++)
                {
                    float weightsDelta = gamma[i] * _inputs[j];
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
        public float[] BackPropagateHidden(float[] gammaForward, float[,] weightsForward)
        {
            float[] gamma = new float[_numberOfOutputs];
            for (int i = 0; i < _numberOfOutputs; i++)
            {
                for (int j = 0; j < gammaForward.Length; j++)
                    gamma[i] += gammaForward[j] * weightsForward[j, i];
                gamma[i] *= _activationFunction(_outputs[i]);
                for (int j = 0; j < _numberOfInputs; j++)
                {
                    float weightsDelta = gamma[i] * _inputs[j];
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
        private float UpdateWeights(float weight, float weightsDelta, float learningRate)
        {
            return weight - weightsDelta * learningRate;
        }
    }
}
