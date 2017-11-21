﻿using System;

namespace NeuralNetwork_CSharp.Examples
{
    class Program
    {
        static void Main(string[] args)
        {
            NeuralNetwork nn = new NeuralNetwork(new[] { 3, 20, 10, 1 });
            for (int i = 0; i < 1000; i++)
            {
                nn.Forward(new float[] { 0, 0, 0 });
                nn.BackPropagate(new float[] { 0 });

                nn.Forward(new float[] { 0, 0, 1 });
                nn.BackPropagate(new float[] { 1 });

                nn.Forward(new float[] { 0, 1, 0 });
                nn.BackPropagate(new float[] { 1 });

                nn.Forward(new float[] { 0, 1, 1 });
                nn.BackPropagate(new float[] { 0 });

                nn.Forward(new float[] { 1, 0, 0 });
                nn.BackPropagate(new float[] { 1 });

                nn.Forward(new float[] { 1, 0, 1 });
                nn.BackPropagate(new float[] { 0 });

                nn.Forward(new float[] { 1, 1, 0 });
                nn.BackPropagate(new float[] { 0 });

                nn.Forward(new float[] { 1, 1, 1 });
                nn.BackPropagate(new float[] { 1 });
            }

            Console.WriteLine($"Expected: 0 | Output: {nn.Forward(new float[] { 0, 0, 0 })[0]}");
            Console.WriteLine($"Expected: 1 | Output: {nn.Forward(new float[] { 0, 0, 1 })[0]}");
            Console.WriteLine($"Expected: 1 | Output: {nn.Forward(new float[] { 0, 1, 0 })[0]}");
            Console.WriteLine($"Expected: 0 | Output: {nn.Forward(new float[] { 0, 1, 1 })[0]}");
            Console.WriteLine($"Expected: 1 | Output: {nn.Forward(new float[] { 1, 0, 0 })[0]}");
            Console.WriteLine($"Expected: 0 | Output: {nn.Forward(new float[] { 1, 0, 1 })[0]}");
            Console.WriteLine($"Expected: 0 | Output: {nn.Forward(new float[] { 1, 1, 0 })[0]}");
            Console.WriteLine($"Expected: 1 | Output: {nn.Forward(new float[] { 1, 1, 1 })[0]}");

            Console.ReadLine();
        }
    }
}
