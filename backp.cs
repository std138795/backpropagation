using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http.Headers;
using System.Reflection.Metadata.Ecma335;
using System.Text;
using System.Threading.Tasks;
using static backprobagation.neuron;

namespace backprobagation
{
    public class backp
    {
    }

    //theoroume oti einai  dense dhladh sundeontai oloi me olous
    public class neuron
    {
        public enum activation
        {
            sigmoid,
            relu,
            linear,
        }
        public enum nueron_type
        {
            input,
            hidden,
            output
        }

        public double x0;
        public double w0;
        public double w0_new;
        public double delta_w0;
        //bias x=1
        //threashold x=-1

        //public nueron_type type;
        public string name;
        public bool have_predefined_values;
        public activation activation_type;
        public List<double> weights;// ta barh pou pane ston neurona
        public List<double> weights_new;// ta barh pou pane ston neurona
        public List<double> input_values;//an einai neuronas eisodou exei input values
        public List<double> outputs;//exoun oloi kai prepei na einai osoi einai oi neurones tou epomenou epipedou
        public double slope;//klish mono se output neurons and hidden neurons
        //TODO
        public double momentum;//prepei na exoume krathsei ta delta ths prohgoumenhs epoxhs
        public double error;//gia tous output neurons
        public List<double> delta_w;
        public double u;

        public double activation_function(double data)
        {

            double ret = 0;
            switch (this.activation_type)
            {
                case activation.sigmoid:
                    ret = sigmoid(data);
                    break;
                case activation.relu:
                    ret = relu(data);
                    break;
                default:
                    ret = lienar(data);
                    break;
            }
            return ret;
        }

        public double sigmoid(double data)
        {
            double k = Math.Exp(data);
            return k / (1.0f + k);
        }
        public double relu(double data)
        {
            return Math.Max(0, data);// x < 0 ? 0 : x;
        }

        public double lienar(double data)
        {
            return data;
        }
        //direvatives

        public double sigmoid_direvative(double data)
        {
            return sigmoid(data) * (1 - sigmoid(data));
        }
        public double relu_direvative(double data)
        {
            return data < 0 ? 0 : 1;
        }

        public double lienar_direvative()
        {
            return 1;
        }
    }

    public class layer
    {
        public double learning_rate = 1;
        public List<neuron> neurons;
        public nueron_type type;
        public uint order;
        public layer(uint _order)
        {
            if (_order == 0)
            {
                type = nueron_type.input;
            }
            order = _order;
        }
    }

    public class traindata
    {
        public List<double> input_values;
        public List<double> outpouts;
        public List<List<double>> outpouts_new;
        public double error_to_stop = 0.01;
    }

    public class Neural
    {
        public int max_epoches = 10000;
        public Neural(List<layer> _layers)
        {
            layers = _layers;
            Init();
        }
        public List<traindata> traindata = new List<traindata>();
        public List<layer> layers;
        public void Init()
        {
            //must check tha all layers have neurons and neurons have weights

            if (layers.Count == 0)
                throw new Exception("Add layers");
            if (layers.Any(x => x.neurons == null || x.neurons.Count == 0))
                throw new Exception("Add neurons to layers");
            if (!layers.Any(x => x.type == nueron_type.output))
                throw new Exception("Add output layer");
            if (!layers.Any(x => x.type == nueron_type.input))
                throw new Exception("Add input layer");
            if (!layers.Any(x => x.type == nueron_type.hidden))
                throw new Exception("Add hidden layer");
            if (layers.Count(x => x.type == nueron_type.input) > 1)
                throw new Exception("Only one input layer is allowed");
            if (layers.Count(x => x.type == nueron_type.output) > 1)
                throw new Exception("Only one input output is allowed");
            layers = layers.OrderBy(x => x.order).ToList();
            for (int i = 0; i < layers.Count; i++)
            {
                for (int j = 0; j < layers[i].neurons.Count; j++)
                {
                    if (i <= layers.Count - 2)//bale tosa weights osa kai ta neurons tou epomenou layer
                    {
                        if (!layers[i].neurons[j].have_predefined_values)
                        {
                            layers[i].neurons[j].weights = new List<double>(layers[i + 1].neurons.Count);
                            layers[i].neurons[j].weights_new = new List<double>(layers[i + 1].neurons.Count);
                            layers[i].neurons[j].delta_w = new List<double>(layers[i + 1].neurons.Count);
                            layers[i].neurons[j].outputs = new List<double>(layers[i + 1].neurons.Count);

                            for (int k = 0; k < layers[i + 1].neurons.Count; k++)//bale ta weights = 0
                            {
                                layers[i].neurons[j].weights.Add(0);
                                layers[i].neurons[j].delta_w.Add(0);
                                layers[i].neurons[j].outputs.Add(0);
                                layers[i].neurons[j].weights_new.Add(0);
                            }
                        }
                    }
                    else
                    {
                        if (!layers[i].neurons[j].have_predefined_values)
                        {
                            layers[i].neurons[j].weights = new List<double>(layers[i - 1].neurons.Count);
                            layers[i].neurons[j].weights_new = new List<double>(layers[i - 1].neurons.Count);
                            layers[i].neurons[j].delta_w = new List<double>(layers[i - 1].neurons.Count);
                            layers[i].neurons[j].outputs = new List<double>();
                            for (int k = 0; k < layers[i - 1].neurons.Count; k++)//bale ta weights = 0
                            {
                                layers[i].neurons[j].weights.Add(0);
                                layers[i].neurons[j].delta_w.Add(0);
                                layers[i].neurons[j].weights_new.Add(0);
                            }
                            layers[i].neurons[j].outputs.Add(0);
                        }

                    }
                    layers[i].neurons[j].name = $"n_{i}_{j}";
                }
            }
        }

        private void FillNewValues(int train_row_index)
        {
            for (int layer_index = 0; layer_index < layers.Count; layer_index++)
            {
                for (int neuron_index = 0; neuron_index < layers[layer_index].neurons.Count; neuron_index++)
                {
                    layers[layer_index].neurons[neuron_index].weights = layers[layer_index].neurons[neuron_index].weights_new;
                    layers[layer_index].neurons[neuron_index].w0 = layers[layer_index].neurons[neuron_index].w0_new;
                    //clear outputs
                    for (int output_index = 0; output_index < layers[layer_index].neurons[neuron_index].outputs.Count; output_index++)
                    {
                        layers[layer_index].neurons[neuron_index].outputs[output_index] = 0;
                    }
                }
            }
        }

        private void Backward()
        {
            //var hiddenlayers = layers.Where(x => x.type != nueron_type.input).OrderByDescending(x => x.order).ToList();
            var hiddenlayers = layers.OrderByDescending(x => x.order).ToList();

            for (int hidden_layer_index = 0; hidden_layer_index < hiddenlayers.Count; hidden_layer_index++)//gia kathe layer
            {
                for (int hidden_neuron_index = 0; hidden_neuron_index < hiddenlayers[hidden_layer_index].neurons.Count; hidden_neuron_index++)//gia kathe neuronas tou layer
                {
                    var prev_layer = layers.FirstOrDefault(x => x.order == hiddenlayers[hidden_layer_index].order - 1);
                    if (prev_layer != null)
                    {
                        for (int prev_layer_neuron_index = 0; prev_layer_neuron_index < prev_layer.neurons.Count; prev_layer_neuron_index++)//gia kathe neuronas tou proigoumenou layer
                        {
                            hiddenlayers[hidden_layer_index].neurons[hidden_neuron_index].delta_w[prev_layer_neuron_index] =
                                hiddenlayers[hidden_layer_index].learning_rate * hiddenlayers[hidden_layer_index].neurons[hidden_neuron_index].slope *
                                prev_layer.neurons[prev_layer_neuron_index].outputs[hidden_layer_index];
                            hiddenlayers[hidden_layer_index].neurons[hidden_neuron_index].weights_new[prev_layer_neuron_index] =
                                hiddenlayers[hidden_layer_index].neurons[hidden_neuron_index].weights[prev_layer_neuron_index] +
                                hiddenlayers[hidden_layer_index].neurons[hidden_neuron_index].delta_w[prev_layer_neuron_index];
                        }

                    }

                    hiddenlayers[hidden_layer_index].neurons[hidden_neuron_index].delta_w0 = hiddenlayers[hidden_layer_index].learning_rate *
                        hiddenlayers[hidden_layer_index].neurons[hidden_neuron_index].slope *
                                hiddenlayers[hidden_layer_index].neurons[hidden_neuron_index].x0;

                    hiddenlayers[hidden_layer_index].neurons[hidden_neuron_index].w0_new = hiddenlayers[hidden_layer_index].neurons[hidden_neuron_index].w0 +
                                hiddenlayers[hidden_layer_index].neurons[hidden_neuron_index].delta_w0;
                }
            }
        }

        private void Forward(int train_row_index, int retrain)
        {
            #region input neuron
            var inputlayer = layers.FirstOrDefault(x => x.type == nueron_type.input);
            for (int train_value_index = 0; train_value_index < traindata[train_row_index].input_values.Count; train_value_index++)//pare tis times kai baltes to proto layer se kathe neurona me th seira
            {
                inputlayer.neurons[train_value_index].input_values = new List<double>();
                //einai neuronas eisodou ara bazoume mia timh
                inputlayer.neurons[train_value_index].input_values.Add(traindata[train_row_index].input_values[train_value_index]);

                //gia kathe neurona eisodou upologhse to weight
                if (!inputlayer.neurons[train_value_index].have_predefined_values)
                {
                    for (int weight_index = 0; weight_index < inputlayer.neurons[train_value_index].weights.Count; weight_index++)
                    {
                        inputlayer.neurons[train_value_index].weights[weight_index] = create_weight(inputlayer.neurons[train_value_index].weights[weight_index]);
                    }
                }


                //gia kathe neurona eisodou upologhse to u pou tha mpei
                for (int output_index = 0; output_index < inputlayer.neurons[train_value_index].outputs.Count; output_index++)
                {
                    inputlayer.neurons[train_value_index].outputs[output_index] = inputlayer.neurons[train_value_index].input_values[0];// * inputlayer.neurons[train_value_index].weights[output_index];
                }
            }
            #endregion

            #region layers for hidden neurons

            var hiddenlayers = layers.Where(x => x.type != nueron_type.input).OrderBy(x => x.order).ToList();


            for (int hidden_layer_index = 0; hidden_layer_index < hiddenlayers.Count; hidden_layer_index++)//gia kathe layer
            {
                for (int hidden_neuron_index = 0; hidden_neuron_index < hiddenlayers[hidden_layer_index].neurons.Count; hidden_neuron_index++)//gia kathe neuronas tou layer
                {
                    //pare ta outputs tou prohgoumenou layer
                    var prev_layer = layers.FirstOrDefault(x => x.order == hiddenlayers[hidden_layer_index].order - 1);
                    var prev_layer_outputs = prev_layer.neurons.Select(x => x.outputs[hidden_neuron_index]).ToList();

                    //bale ta sthn input values tou neuronas
                    hiddenlayers[hidden_layer_index].neurons[hidden_neuron_index].input_values = prev_layer_outputs;
                    //gia kathe neurona eisodou upologhse to weight
                    if (!hiddenlayers[hidden_layer_index].neurons[hidden_neuron_index].have_predefined_values)
                    {
                        for (int weight_index = 0; weight_index < hiddenlayers[hidden_layer_index].neurons[hidden_neuron_index].weights.Count; weight_index++)
                        {
                            hiddenlayers[hidden_layer_index].neurons[hidden_neuron_index].weights[weight_index] = create_weight(hiddenlayers[hidden_layer_index].neurons[hidden_neuron_index].weights[weight_index]);
                        }

                        hiddenlayers[hidden_layer_index].neurons[hidden_neuron_index].w0 = create_weight(hiddenlayers[hidden_layer_index].neurons[hidden_neuron_index].w0);
                    }

                    //gia kathe neurona eisodou upologhse to u pou tha mpei
                    for (int output_index = 0; output_index < hiddenlayers[hidden_layer_index].neurons[hidden_neuron_index].outputs.Count; output_index++)
                    {
                        for (int input_index = 0; input_index < hiddenlayers[hidden_layer_index].neurons[hidden_neuron_index].input_values.Count; input_index++)
                        {
                            hiddenlayers[hidden_layer_index].neurons[hidden_neuron_index].outputs[output_index] +=
                            hiddenlayers[hidden_layer_index].neurons[hidden_neuron_index].input_values[input_index] *
                            hiddenlayers[hidden_layer_index].neurons[hidden_neuron_index].weights[output_index];
                        }

                        hiddenlayers[hidden_layer_index].neurons[hidden_neuron_index].outputs[output_index] +=
                            hiddenlayers[hidden_layer_index].neurons[hidden_neuron_index].x0 * hiddenlayers[hidden_layer_index].neurons[hidden_neuron_index].w0;
                        hiddenlayers[hidden_layer_index].neurons[hidden_neuron_index].u = hiddenlayers[hidden_layer_index].neurons[hidden_neuron_index].outputs[output_index];
                        //perna apo activation function to u
                        hiddenlayers[hidden_layer_index].neurons[hidden_neuron_index].outputs[output_index] =
                            hiddenlayers[hidden_layer_index].neurons[hidden_neuron_index].activation_function(hiddenlayers[hidden_layer_index].neurons[hidden_neuron_index].u);

                    }
                }
            }

            //bres to output layer
            var outneurons = layers.Where(x => x.type == nueron_type.output).OrderBy(x => x.order).FirstOrDefault().neurons;
            bool have_error = false;
            for (int neuron_index = 0; neuron_index < outneurons.Count; neuron_index++)
            {
                var error = traindata[train_row_index].outpouts[neuron_index] - outneurons[neuron_index].outputs.FirstOrDefault();
                if (error != 0)
                {
                    if (traindata[train_row_index].outpouts_new.Count == 0 ||
                        traindata[train_row_index].outpouts_new.Count <= retrain ||
                        traindata[train_row_index].outpouts_new[retrain].Count == 0)
                    {
                        traindata[train_row_index].outpouts_new.Add(new List<double> { outneurons[neuron_index].outputs.FirstOrDefault() });
                    }
                    else
                    {
                        traindata[train_row_index].outpouts_new[retrain].Add(outneurons[neuron_index].outputs.FirstOrDefault());
                    }
                    have_error = true;
                    outneurons[neuron_index].error = error;
                    //calculate slope
                    outneurons[neuron_index].slope = error * outneurons[neuron_index].sigmoid_direvative(outneurons[neuron_index].u);
                    //init deltas
                    outneurons[neuron_index].delta_w = new List<double>();
                    for (int weight_index = 0; weight_index < outneurons[neuron_index].weights.Count; weight_index++)
                    {
                        outneurons[neuron_index].delta_w.Add(0);
                    }
                }
            }

            if (have_error)
            {
                //bres ta hidden layers
                var hiddenlayers2 = layers.Where(x => x.type == nueron_type.hidden).OrderBy(x => x.order).ToList();
                for (int layer_index = 0; layer_index < hiddenlayers2.Count; layer_index++)
                {
                    for (int neuron_index = 0; neuron_index < hiddenlayers2[layer_index].neurons.Count; neuron_index++)
                    {
                        //get slope and weignts from next layer
                        var next_layer = layers.FirstOrDefault(x => x.order == hiddenlayers2[layer_index].order + 1);
                        if (next_layer != null)
                        {
                            for (int next_neuron_index = 0; next_neuron_index < next_layer.neurons.Count; next_neuron_index++)
                            {
                                hiddenlayers2[layer_index].neurons[neuron_index].slope +=
                                    next_layer.neurons[next_neuron_index].slope * next_layer.neurons[next_neuron_index].weights[neuron_index];
                                //hidden layer slope = next layer slope * next layer weight indexed from current neuron
                            }
                            hiddenlayers2[layer_index].neurons[neuron_index].slope = hiddenlayers2[layer_index].neurons[neuron_index].sigmoid_direvative(hiddenlayers2[layer_index].neurons[neuron_index].u) * hiddenlayers2[layer_index].neurons[neuron_index].slope;
                        }
                        //init deltas
                        hiddenlayers2[layer_index].neurons[neuron_index].delta_w = new List<double>();
                        for (int weight_index = 0; weight_index < hiddenlayers2[layer_index].neurons[neuron_index].weights.Count; weight_index++)
                        {
                            hiddenlayers2[layer_index].neurons[neuron_index].delta_w.Add(0);
                        }
                    }
                }
            }
            #endregion

        }

        public void Train(int epohi)
        {

            if (traindata == null || traindata.Count == 0)
                throw new Exception("Add train data");
            if (traindata.Any(x => x == null || x.input_values == null || x.input_values?.Count == 0 ||
            x.input_values?.Count != layers?.OrderBy(x => x.order)?.FirstOrDefault(x => x.type == neuron.nueron_type.input)?.neurons.Count))
                throw new Exception("train data inputs number must be equal with inputs");


            //pare ta input values kai bale ta sthn prwth layer
            while (epohi <= max_epoches)
            {
                for (int train_row_index = 0; train_row_index < traindata.Count; train_row_index++)//gia kathe row tou pinaka alhtheias
                {
                    if (traindata[train_row_index].outpouts_new == null)
                    {
                        traindata[train_row_index].outpouts_new = new List<List<double>>();
                    }
                    Console.WriteLine($"Start Forward in train data row {train_row_index} and epohi {epohi}");
                    Forward(train_row_index, epohi);
                    Console.WriteLine($"Start Backward in train data row {train_row_index} and epohi {epohi}");
                    Backward();
                    Console.WriteLine($"Clear outputs in train data row {train_row_index} and epohi {epohi}");
                    FillNewValues(train_row_index);

                    epohi++;
                    if (StopTrainThisRow(train_row_index))
                    {
                        epohi = max_epoches + 1;
                    }
                }
            }


        }

        public bool StopTrainThisRow(int train_row_index)
        {
            bool ret = false;
            var last_outs = traindata[train_row_index].outpouts_new.LastOrDefault();
            var prevlast_outs = new List<double>();
            if (traindata[train_row_index].outpouts_new.Count - 2 >= 0)
                prevlast_outs = traindata[train_row_index].outpouts_new[traindata[train_row_index].outpouts_new.Count - 2];
            //ama exoun mikrh diafora increase learning rate
            for (int out_index = 0; out_index < traindata[train_row_index].outpouts.Count; out_index++)
            {
                if (last_outs[out_index] != traindata[train_row_index].outpouts[out_index])
                {
                    if (last_outs[out_index] > traindata[train_row_index].outpouts[out_index])
                    {
                        var diff = last_outs[out_index] - traindata[train_row_index].outpouts[out_index];
                        if (diff < traindata[train_row_index].error_to_stop)
                            ret = true;
                        if (last_outs.SequenceEqual(prevlast_outs))
                            ret = true;
                        break;
                    }
                    else
                    {
                        var diff = traindata[train_row_index].outpouts[out_index] - last_outs[out_index];
                        if (diff < traindata[train_row_index].error_to_stop)
                            ret = true;
                        if (last_outs.SequenceEqual(prevlast_outs))
                            ret = true;
                        break;
                    }
                }
            }

            return ret;
        }

        private void CompareOutPutValues(int train_row_index)
        {
            bool result = true;
            var last_outs = traindata[train_row_index].outpouts_new.LastOrDefault();
            //ama exoun mikrh diafora increase learning rate
            for (int out_index = 0; out_index < traindata[train_row_index].outpouts.Count; out_index++)
            {
                if (last_outs[out_index] != traindata[train_row_index].outpouts[out_index])
                {

                }
            }
        }

        private double create_weight(double old_weight)
        {
            Random rnd = new Random();
            double new_weight = old_weight + rnd.NextDouble() * 0.1 - 0.05;
            return new_weight;
        }

    }
}
