using backprobagation;
using System;
namespace CodingBackProp
{
    class BackPropProgram
    {
        static void Main(string[] args)
        {
            List<layer> layers = new List<layer>();
            layer input_layer = new layer(0);
            input_layer.neurons = new List<neuron>();
            input_layer.type = neuron.nueron_type.input;
            //2 eisodous
            input_layer.neurons.Add(new neuron()
            {
                momentum = 0.01,
                //have_predefined_values = false,
                //outputs = new List<double> { 0, 0, 0 },
                //weights = new List<double> { 0.3, 0.3, 0.3 }

            });
            input_layer.neurons.Add(new neuron()
            {
                momentum = 0.01,
                //have_predefined_values = true,
                //outputs = new List<double> { 0, 0, 0 },
                //weights = new List<double> { 0.3, 0.3, 0.3 }
            });
            //2 hidden
            layer hidden_layer = new layer(1);
            hidden_layer.neurons = new List<neuron>();
            hidden_layer.type = neuron.nueron_type.hidden;

            hidden_layer.neurons.Add(new neuron()
            {
                activation_type = neuron.activation.sigmoid,
                //have_predefined_values = true,
                //outputs = new List<double> { 0, 0 },
                //weights = new List<double> { 0, 0, 0 },
                ////weights = new List<double> { 0.3, 0.3, 0.3 },
                //weights_new = new List<double> { 0, 0, 0 },
                momentum = 0.01,
                x0 = -1,
                w0 = 0.5
            });
            hidden_layer.neurons.Add(new neuron()
            {
                activation_type = neuron.activation.sigmoid,
                momentum = 0.01,
                x0 = -1,
                w0 = 0.5,
                //have_predefined_values = true,
                ////weights = new List<double> { 0.3, 0.3, 0.3 },
                //weights = new List<double> { 0, 0, 0 },
                //weights_new = new List<double> { 0, 0, 0 },
                //outputs = new List<double> { 0, 0 },
            });
            hidden_layer.neurons.Add(new neuron()
            {
                activation_type = neuron.activation.sigmoid,
                momentum = 0.01,
                x0 = -1,
                w0 = 0.5,
                //have_predefined_values = true,
                //outputs = new List<double> { 0, 0 },
                ////weights = new List<double> { 0.3, 0.3, 0.3 },
                //weights = new List<double> { 0, 0, 0 },
                //weights_new = new List<double> { 0, 0, 0 },
            });
            //2 output
            layer output_layer = new layer(2);
            output_layer.type = neuron.nueron_type.output;
            output_layer.neurons = new List<neuron>();
            output_layer.neurons.Add(new neuron()
            {
                activation_type = neuron.activation.sigmoid,
                momentum = 0.01,
                x0 = -1,
                w0 = 0.4,
                //have_predefined_values = true,
                //outputs = new List<double> { 0 },
                ////weights = new List<double> { 0.2, 0.2, 0.2 },
                //weights = new List<double> { 0, 0, 0 },
                //weights_new = new List<double> { 0, 0, 0 },
            });
            output_layer.neurons.Add(new neuron()
            {
                activation_type = neuron.activation.sigmoid,
                momentum = 0.01,
                x0 = -1,
                w0 = 0.4,
                //have_predefined_values = true,
                //outputs = new List<double> { 0 },
                ////weights = new List<double> { 0.3, 0.3, 0.3 },
                //weights = new List<double> { 0, 0, 0 },
                //weights_new = new List<double> { 0, 0, 0 },
            });
            layers.Add(input_layer);
            layers.Add(hidden_layer);
            layers.Add(output_layer);
            Neural backp = new Neural(layers);
            backp.Init();
            backp.traindata = new List<traindata>();
            backp.traindata.Add(new traindata
            {
                input_values = new List<double>() { 0.5, 0.5 },
                outpouts = new List<double> { 1, 0 }
            });

            backp.Train(0);

            //backp.traindata.Add(new traindata
            //{
            //    input_values = new List<double>() { 0, 1 },
            //    outpout = 0

            //});
            //backp.traindata.Add(new traindata
            //{
            //    input_values = new List<double>() { 1, 0 },
            //    outpout = 0

            //});

            //backp.traindata.Add(new traindata
            //{
            //    input_values = new List<double>() { 1, 1 },
            //    outpout = 1

            //});
            Console.ReadLine();
        } // Main
    } // Program

} // ns
