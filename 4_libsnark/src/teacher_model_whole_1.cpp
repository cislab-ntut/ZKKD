#include <libsnark/common/default_types/r1cs_gg_ppzksnark_pp.hpp>
#include <libsnark/zk_proof_systems/ppzksnark/r1cs_gg_ppzksnark/r1cs_gg_ppzksnark.hpp>
#include <libsnark/gadgetlib1/pb_variable.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <nlohmann/json.hpp>
#include <unordered_map>
#include <queue>
#include <sys/time.h>

using namespace libsnark;
using namespace std;
using json = nlohmann::json;

typedef libff::Fr<default_r1cs_gg_ppzksnark_pp> FieldT;
static int SCALE_FACTOR = 1024; // 2^10

uint64_t ROOT_ID;

struct FC
{
    string layer;
    vector<int> input;
    vector<int> linear_output;
    vector<vector<int>> layer_weight;
    vector<int> layer_bias;
};

struct ReLU
{
    string layer;
    vector<int> input;
    vector<int> output;
    vector<int> relu_operator;
};

void to_json(json &j, const FC &fc)
{
    j = json{
        {"layer", fc.layer},
        {"input", fc.input},
        {"linear_output", fc.linear_output},
        {"layer_weight", fc.layer_weight},
        {"layer_bias", fc.layer_bias}};
}

void from_json(const json &j, FC &fc)
{
    fc.layer = j.at("layer").get<string>();
    fc.input = j.at("input").get<vector<int>>();
    fc.linear_output = j.at("linear_output").get<vector<int>>();
    fc.layer_weight = j.at("layer_weight").get<vector<vector<int>>>();
    fc.layer_bias = j.at("layer_bias").get<vector<int>>();
}

void to_json(json &j, const ReLU &relu)
{
    j = json{
        {"layer", relu.layer},
        {"input", relu.input},
        {"output", relu.output},
        {"relu_operator", relu.relu_operator}};
}

void from_json(const json &j, ReLU &relu)
{
    relu.layer = j.at("layer").get<string>();
    relu.input = j.at("input").get<vector<int>>();
    relu.output = j.at("output").get<vector<int>>();
    relu.relu_operator = j.at("relu_operator").get<vector<int>>();
}

int python_mod(int a, int b)
{
    int r = a % b;
    return r < 0 ? r + abs(b) : r;
}

struct ReadFCParams
{
    vector<json> operation_logs;

    void read_operation_logs()
    {
        const vector<string> paths = {
            "extraction_teacher_whole/fc1/fc1_operation_log.json",
            "extraction_teacher_whole/fc2/fc2_operation_log.json",
            "extraction_teacher_whole/fc3/fc3_operation_log.json"};

        for (const auto &path : paths)
        {
            ifstream file(path);
            if (!file)
            {
                cerr << "Unable to open file: " << path << endl;
                continue;
            }

            json operation_log;
            file >> operation_log;
            operation_logs.push_back(operation_log);
        }
    }
};

struct ReadReLUParams
{
    vector<json> operation_logs;

    void read_operation_logs()
    {
        const vector<string> paths = {
            "extraction_teacher_whole/relu1/relu1_operation_log.json",
            "extraction_teacher_whole/relu2/relu2_operation_log.json"};

        for (const auto &path : paths)
        {
            ifstream file(path);
            if (!file)
            {
                cerr << "Unable to open file: " << path << endl;
                continue;
            }

            json operation_log;
            file >> operation_log;
            operation_logs.push_back(operation_log);
        }
    }
};

int main()
{
    // fc1 + relu
    // initialize the curve parameters
    default_r1cs_gg_ppzksnark_pp::init_public_params();
    protoboard<FieldT> pb;

    ReadFCParams fc_read;
    fc_read.read_operation_logs();

    FC fc1 = fc_read.operation_logs[0];
    FC fc2 = fc_read.operation_logs[1];
    FC fc3 = fc_read.operation_logs[2];


    ///////////////////////////// fc1 /////////////////////////////
    vector<pb_variable<FieldT>> input(784);
    vector<vector<pb_variable<FieldT>>> fc1_weight(1200, vector<pb_variable<FieldT>>(784));
    vector<pb_variable<FieldT>> fc1_bias(1200);
    vector<pb_variable<FieldT>> fc1_linear_output(1200);

    for (int input_index = 0; input_index < 784; input_index++)
    {
        string name = "input_" + to_string(input_index) + fc1.layer;
        input[input_index].allocate(pb, name);
        pb.val(input[input_index]) = fc1.input[input_index];
    }

    for (int weight_index = 0; weight_index < 1200; weight_index++)
    {
        for (int input_index = 0; input_index < 784; input_index++)
        {
            string name = "fc1_weight_" + to_string(weight_index) + "_" + to_string(input_index);
            fc1_weight[weight_index][input_index].allocate(pb, name);
            pb.val(fc1_weight[weight_index][input_index]) = fc1.layer_weight[weight_index][input_index];
        }
    }

    for (int bias_index = 0; bias_index < 1200; bias_index++)
    {
        string name = "fc1_bias_" + to_string(bias_index);
        fc1_bias[bias_index].allocate(pb, name);
        pb.val(fc1_bias[bias_index]) = fc1.layer_bias[bias_index];
    }

    for (int linear_output_index = 0; linear_output_index < 1200; linear_output_index++)
    {
        string name = "fc1_linear_output_" + to_string(linear_output_index);
        fc1_linear_output[linear_output_index].allocate(pb, name);
        pb.val(fc1_linear_output[linear_output_index]) = fc1.linear_output[linear_output_index];
    }

    int arr_fc1_weight_mul_temp[1200][784];
    vector<vector<pb_variable<FieldT>>> fc1_weight_mul_temp(1200, vector<pb_variable<FieldT>>(784));

    for (int weight_index = 0; weight_index < 1200; weight_index++)
    {
        for (int mul_index = 0; mul_index < 784; mul_index++)
        {
            arr_fc1_weight_mul_temp[weight_index][mul_index] = fc1.layer_weight[weight_index][mul_index] * fc1.input[mul_index];
            string name = "fc1_weight_mul_temp_" + to_string(weight_index) + "_" + to_string(mul_index);
            fc1_weight_mul_temp[weight_index][mul_index].allocate(pb, name);
            pb.val(fc1_weight_mul_temp[weight_index][mul_index]) = arr_fc1_weight_mul_temp[weight_index][mul_index];
            pb.add_r1cs_constraint(r1cs_constraint<FieldT>(input[mul_index], fc1_weight[weight_index][mul_index], fc1_weight_mul_temp[weight_index][mul_index]));
        }
    }

    int arr_fc1_weight_add_temp[1200][783];
    vector<vector<pb_variable<FieldT>>> fc1_weight_add_temp(1200, vector<pb_variable<FieldT>>(783));

    for (int weight_index = 0; weight_index < 1200; weight_index++)
    {
        for (int fc_weight_index = 0; fc_weight_index < 783; fc_weight_index++)
        {
            if (fc_weight_index == 0)
            {
                arr_fc1_weight_add_temp[weight_index][fc_weight_index] = arr_fc1_weight_mul_temp[weight_index][fc_weight_index] + arr_fc1_weight_mul_temp[weight_index][fc_weight_index + 1];
            }
            else
            {
                arr_fc1_weight_add_temp[weight_index][fc_weight_index] = arr_fc1_weight_add_temp[weight_index][fc_weight_index - 1] + arr_fc1_weight_mul_temp[weight_index][fc_weight_index + 1];
            }

            string name = "fc1_weight_add_temp_" + to_string(weight_index) + "_" + to_string(fc_weight_index);
            fc1_weight_add_temp[weight_index][fc_weight_index].allocate(pb, name);
            pb.val(fc1_weight_add_temp[weight_index][fc_weight_index]) = arr_fc1_weight_add_temp[weight_index][fc_weight_index];
        }
    }

    for (int weight_index = 0; weight_index < 1200; weight_index++)
    {
        for (int fc_weight_index = 0; fc_weight_index < 783; fc_weight_index++)
        {
            if (fc_weight_index == 0)
            {
                pb.add_r1cs_constraint(r1cs_constraint<FieldT>(fc1_weight_mul_temp[weight_index][fc_weight_index] + fc1_weight_mul_temp[weight_index][fc_weight_index + 1], 1, fc1_weight_add_temp[weight_index][fc_weight_index]));
            }
            else
            {
                pb.add_r1cs_constraint(r1cs_constraint<FieldT>(fc1_weight_add_temp[weight_index][fc_weight_index - 1] + fc1_weight_mul_temp[weight_index][fc_weight_index + 1], 1, fc1_weight_add_temp[weight_index][fc_weight_index]));
            }
        }
    }

    // descale
    pb_variable<FieldT> fc1_weight_add_temp_rescale_mod[1200];
    for (int weight_index = 0; weight_index < 1200; weight_index++)
    {
        string name = "fc1_weight_add_temp_rescale_mod_" + to_string(weight_index);
        fc1_weight_add_temp_rescale_mod[weight_index].allocate(pb, name);
        pb.val(fc1_weight_add_temp_rescale_mod[weight_index]) = python_mod(arr_fc1_weight_add_temp[weight_index][782], SCALE_FACTOR);
        pb.add_r1cs_constraint(r1cs_constraint<FieldT>(fc1_linear_output[weight_index] - fc1_bias[weight_index], SCALE_FACTOR, fc1_weight_add_temp[weight_index][782] - fc1_weight_add_temp_rescale_mod[weight_index]));
    }

    ///////////////////////////// fc2 /////////////////////////////
    // vector<pb_variable<FieldT>> fc2_input(1200);
    // vector<vector<pb_variable<FieldT>>> fc2_weight(1200, vector<pb_variable<FieldT>>(1200));
    // vector<pb_variable<FieldT>> fc2_bias(1200);
    // vector<pb_variable<FieldT>> fc2_linear_output(1200);

    // for (int input_index = 0; input_index < 1200; input_index++)
    // {
    //     string name = "input_" + to_string(input_index) + fc2.layer;
    //     fc2_input[input_index].allocate(pb, name);
    //     pb.val(fc2_input[input_index]) = fc2.input[input_index];
    // }

    // for (int weight_index = 0; weight_index < 1200; weight_index++)
    // {
    //     for (int input_index = 0; input_index < 1200; input_index++)
    //     {
    //         string name = "fc2_weight_" + to_string(weight_index) + "_" + to_string(input_index);
    //         fc2_weight[weight_index][input_index].allocate(pb, name);
    //         pb.val(fc2_weight[weight_index][input_index]) = fc2.layer_weight[weight_index][input_index];
    //     }
    // }

    // for (int bias_index = 0; bias_index < 1200; bias_index++)
    // {
    //     string name = "fc2_bias_" + to_string(bias_index);
    //     fc2_bias[bias_index].allocate(pb, name);
    //     pb.val(fc2_bias[bias_index]) = fc2.layer_bias[bias_index];
    // }

    // for (int linear_output_index = 0; linear_output_index < 1200; linear_output_index++)
    // {
    //     string name = "fc2_linear_output_" + to_string(linear_output_index);
    //     fc2_linear_output[linear_output_index].allocate(pb, name);
    //     pb.val(fc2_linear_output[linear_output_index]) = fc2.linear_output[linear_output_index];
    // }

    // int arr_fc2_weight_mul_temp[1200][1200];
    // vector<vector<pb_variable<FieldT>>> fc2_weight_mul_temp(1200, vector<pb_variable<FieldT>>(1200));

    // for (int weight_index = 0; weight_index < 1200; weight_index++)
    // {
    //     for (int mul_index = 0; mul_index < 1200; mul_index++)
    //     {
    //         arr_fc2_weight_mul_temp[weight_index][mul_index] = fc2.layer_weight[weight_index][mul_index] * fc2.input[mul_index];
    //         string name = "fc2_weight_mul_temp_" + to_string(weight_index) + "_" + to_string(mul_index);
    //         fc2_weight_mul_temp[weight_index][mul_index].allocate(pb, name);
    //         pb.val(fc2_weight_mul_temp[weight_index][mul_index]) = arr_fc2_weight_mul_temp[weight_index][mul_index];
    //         pb.add_r1cs_constraint(r1cs_constraint<FieldT>(fc2_input[mul_index], fc2_weight[weight_index][mul_index], fc2_weight_mul_temp[weight_index][mul_index]));
    //     }
    // }

    // int arr_fc2_weight_add_temp[1200][1199];
    // vector<vector<pb_variable<FieldT>>> fc2_weight_add_temp(1200, vector<pb_variable<FieldT>>(1199));

    // for (int weight_index = 0; weight_index < 1200; weight_index++)
    // {
    //     for (int fc_weight_index = 0; fc_weight_index < 1199; fc_weight_index++)
    //     {
    //         if (fc_weight_index == 0)
    //         {
    //             arr_fc2_weight_add_temp[weight_index][fc_weight_index] = arr_fc2_weight_mul_temp[weight_index][fc_weight_index] + arr_fc2_weight_mul_temp[weight_index][fc_weight_index + 1];
    //         }
    //         else
    //         {
    //             arr_fc2_weight_add_temp[weight_index][fc_weight_index] = arr_fc2_weight_add_temp[weight_index][fc_weight_index - 1] + arr_fc2_weight_mul_temp[weight_index][fc_weight_index + 1];
    //         }

    //         string name = "fc2_weight_add_temp_" + to_string(weight_index) + "_" + to_string(fc_weight_index);
    //         fc2_weight_add_temp[weight_index][fc_weight_index].allocate(pb, name);
    //         pb.val(fc2_weight_add_temp[weight_index][fc_weight_index]) = arr_fc2_weight_add_temp[weight_index][fc_weight_index];
    //     }
    // }

    // for (int weight_index = 0; weight_index < 1200; weight_index++){
    //     for (int fc_weight_index = 0; fc_weight_index < 1199; fc_weight_index++){
    //         if (fc_weight_index == 0)
    //         {
    //             pb.add_r1cs_constraint(r1cs_constraint<FieldT>(fc2_weight_mul_temp[weight_index][fc_weight_index] + fc2_weight_mul_temp[weight_index][fc_weight_index + 1], 1, fc2_weight_add_temp[weight_index][fc_weight_index]));
    //         }
    //         else
    //         {
    //             pb.add_r1cs_constraint(r1cs_constraint<FieldT>(fc2_weight_add_temp[weight_index][fc_weight_index - 1] + fc2_weight_mul_temp[weight_index][fc_weight_index + 1], 1, fc2_weight_add_temp[weight_index][fc_weight_index]));
    //         }
    //     }
    // }

    // // descale

    // pb_variable<FieldT> fc2_weight_add_temp_rescale_mod[1200];
    // for (int weight_index = 0; weight_index < 1200; weight_index++)
    // {
    //     string name = "fc2_weight_add_temp_rescale_mod_" + to_string(weight_index);
    //     fc2_weight_add_temp_rescale_mod[weight_index].allocate(pb, name);
    //     pb.val(fc2_weight_add_temp_rescale_mod[weight_index]) = python_mod(arr_fc2_weight_add_temp[weight_index][1199], SCALE_FACTOR);
    //     pb.add_r1cs_constraint(r1cs_constraint<FieldT>(fc2_linear_output[weight_index] - fc2_bias[weight_index], SCALE_FACTOR, fc2_weight_add_temp[weight_index][1199] - fc2_weight_add_temp_rescale_mod[weight_index]));
    // }

    // ///////////////////////////// fc3 /////////////////////////////
    // vector<pb_variable<FieldT>> fc3_input(1200);
    // vector<vector<pb_variable<FieldT>>> fc3_weight(10, vector<pb_variable<FieldT>>(1200));
    // vector<pb_variable<FieldT>> fc3_bias(10);
    // vector<pb_variable<FieldT>> fc3_linear_output(10);

    // for (int input_index = 0; input_index < 1200; input_index++)
    // {
    //     string name = "input_" + to_string(input_index) + fc3.layer;
    //     fc3_input[input_index].allocate(pb, name);
    //     pb.val(fc3_input[input_index]) = fc3.input[input_index];
    // }

    // for (int weight_index = 0; weight_index < 10; weight_index++)
    // {
    //     for (int input_index = 0; input_index < 1200; input_index++)
    //     {
    //         string name = "fc3_weight_" + to_string(weight_index) + "_" + to_string(input_index);
    //         fc3_weight[weight_index][input_index].allocate(pb, name);
    //         pb.val(fc3_weight[weight_index][input_index]) = fc3.layer_weight[weight_index][input_index];
    //     }
    // }

    // for (int bias_index = 0; bias_index < 10; bias_index++)
    // {
    //     string name = "fc3_bias_" + to_string(bias_index);
    //     fc3_bias[bias_index].allocate(pb, name);
    //     pb.val(fc3_bias[bias_index]) = fc3.layer_bias[bias_index];
    // }

    // for (int linear_output_index = 0; linear_output_index < 10; linear_output_index++)
    // {
    //     string name = "fc3_linear_output_" + to_string(linear_output_index);
    //     fc3_linear_output[linear_output_index].allocate(pb, name);
    //     pb.val(fc3_linear_output[linear_output_index]) = fc3.linear_output[linear_output_index];
    // }

    // int arr_fc3_weight_mul_temp[10][1200];
    // vector<vector<pb_variable<FieldT>>> fc3_weight_mul_temp(10, vector<pb_variable<FieldT>>(1200));
    

    // for (int weight_index = 0; weight_index < 10; weight_index++)
    // {
    //     for (int mul_index = 0; mul_index < 1200; mul_index++)
    //     {
    //         arr_fc3_weight_mul_temp[weight_index][mul_index] = fc3.layer_weight[weight_index][mul_index] * fc3.input[mul_index];
    //         string name = "fc3_weight_mul_temp_" + to_string(weight_index) + "_" + to_string(mul_index);
    //         fc3_weight_mul_temp[weight_index][mul_index].allocate(pb, name);
    //         pb.val(fc3_weight_mul_temp[weight_index][mul_index]) = arr_fc3_weight_mul_temp[weight_index][mul_index];
    //         pb.add_r1cs_constraint(r1cs_constraint<FieldT>(fc3_input[mul_index], fc3_weight[weight_index][mul_index], fc3_weight_mul_temp[weight_index][mul_index]));
    //     }
    // }

    // int arr_fc3_weight_add_temp[10][1199];
    // vector<vector<pb_variable<FieldT>>> fc3_weight_add_temp(10, vector<pb_variable<FieldT>>(1199));

    // for (int weight_index = 0; weight_index < 10; weight_index++)
    // {
    //     for (int fc_weight_index = 0; fc_weight_index < 1199; fc_weight_index++)
    //     {
    //         if (fc_weight_index == 0)
    //         {
    //             arr_fc3_weight_add_temp[weight_index][fc_weight_index] = arr_fc3_weight_mul_temp[weight_index][fc_weight_index] + arr_fc3_weight_mul_temp[weight_index][fc_weight_index + 1];
    //         }
    //         else
    //         {
    //             arr_fc3_weight_add_temp[weight_index][fc_weight_index] = arr_fc3_weight_add_temp[weight_index][fc_weight_index - 1] + arr_fc3_weight_mul_temp[weight_index][fc_weight_index + 1];
    //         }

    //         string name = "fc3_weight_add_temp_" + to_string(weight_index) + "_" + to_string(fc_weight_index);
    //         fc3_weight_add_temp[weight_index][fc_weight_index].allocate(pb, name);
    //         pb.val(fc3_weight_add_temp[weight_index][fc_weight_index]) = arr_fc3_weight_add_temp[weight_index][fc_weight_index];
    //     }
    // }

    // for (int weight_index = 0; weight_index < 10; weight_index++)
    // {
    //     for (int fc_weight_index = 0; fc_weight_index < 1199; fc_weight_index++)
    //     {
    //         if (fc_weight_index == 0)
    //         {
    //             pb.add_r1cs_constraint(r1cs_constraint<FieldT>(fc3_weight_mul_temp[weight_index][fc_weight_index] + fc3_weight_mul_temp[weight_index][fc_weight_index + 1], 1, fc3_weight_add_temp[weight_index][fc_weight_index]));
    //         }
    //         else
    //         {
    //             pb.add_r1cs_constraint(r1cs_constraint<FieldT>(fc3_weight_add_temp[weight_index][fc_weight_index - 1] + fc3_weight_mul_temp[weight_index][fc_weight_index + 1], 1, fc3_weight_add_temp[weight_index][fc_weight_index]));
    //         }
    //     }
    // }

    // // descale
    // pb_variable<FieldT> fc3_weight_add_temp_rescale_mod[10];
    // for (int weight_index = 0; weight_index < 10; weight_index++)
    // {
    //     string name = "fc3_weight_add_temp_rescale_mod_" + to_string(weight_index);
    //     fc3_weight_add_temp_rescale_mod[weight_index].allocate(pb, name);
    //     pb.val(fc3_weight_add_temp_rescale_mod[weight_index]) = python_mod(arr_fc3_weight_add_temp[weight_index][1199], SCALE_FACTOR);
    //     pb.add_r1cs_constraint(r1cs_constraint<FieldT>(fc3_linear_output[weight_index] - fc3_bias[weight_index], SCALE_FACTOR, fc3_weight_add_temp[weight_index][1199] - fc3_weight_add_temp_rescale_mod[weight_index]));
    // }

    

    // ReadReLUParams relu_read;
    // relu_read.read_operation_logs();
    // for (size_t i = 0; i < relu_read.operation_logs.size(); i++)
    // {
    //     ReLU current_relu = relu_read.operation_logs[i];
    //     pb_variable<FieldT> input[1200];
    //     pb_variable<FieldT> relu_op[1200];
    //     pb_variable<FieldT> output[1200];

    //     // set input
    //     for (int input_index = 0; input_index < 1200; input_index++)
    //     {
    //         string name = "input_" + to_string(input_index) + current_relu.layer;
    //         input[input_index].allocate(pb, name);
    //         pb.val(input[input_index]) = current_relu.input[input_index];
    //     }

    //     // set relu operator
    //     for (int relu_index = 0; relu_index < 1200; relu_index++)
    //     {
    //         string name = "relu_" + to_string(relu_index) + current_relu.layer;
    //         relu_op[relu_index].allocate(pb, name);
    //         pb.val(relu_op[relu_index]) = current_relu.relu_operator[relu_index];
    //     }

    //     // set output
    //     for (int output_index = 0; output_index < 1200; output_index++)
    //     {
    //         string name = "output_" + to_string(output_index) + current_relu.layer;
    //         output[output_index].allocate(pb, name);
    //         pb.val(output[output_index]) = current_relu.output[output_index];
    //     }

    //     for (int relu_index = 0; relu_index < 1200; relu_index++)
    //     {
    //         pb.add_r1cs_constraint(r1cs_constraint<FieldT>(input[relu_index], relu_op[relu_index], output[relu_index]));
    //     }
    // }
    pb.set_input_sizes(784);
    const r1cs_constraint_system<FieldT> constraint_system = pb.get_constraint_system();

    struct timeval StartSetup, EndSetup, StartGenerateProof, EndGenerateProof, StartVerify, EndVerify;
    unsigned long Setup, Proof, Verify;

    // Generate the keypair
    gettimeofday(&StartSetup, NULL);
    const r1cs_gg_ppzksnark_keypair<default_r1cs_gg_ppzksnark_pp> keypair = r1cs_gg_ppzksnark_generator<default_r1cs_gg_ppzksnark_pp>(constraint_system);
    gettimeofday(&EndSetup, NULL);

    // Generate the proof
    gettimeofday(&StartGenerateProof, NULL);
    const r1cs_gg_ppzksnark_proof<default_r1cs_gg_ppzksnark_pp> proof = r1cs_gg_ppzksnark_prover<default_r1cs_gg_ppzksnark_pp>(keypair.pk, pb.primary_input(), pb.auxiliary_input());
    gettimeofday(&EndGenerateProof, NULL);

    // Verification
    gettimeofday(&StartVerify, NULL);
    bool verified = r1cs_gg_ppzksnark_verifier_strong_IC<default_r1cs_gg_ppzksnark_pp>(keypair.vk, pb.primary_input(), proof);
    gettimeofday(&EndVerify, NULL);

    cout << "Verification status: " << verified << endl;

    Setup = 1000000 * (EndSetup.tv_sec - StartSetup.tv_sec) + EndSetup.tv_usec - StartSetup.tv_usec;
    Proof = 1000000 * (EndGenerateProof.tv_sec - StartGenerateProof.tv_sec) + EndGenerateProof.tv_usec - StartGenerateProof.tv_usec;
    Verify = 1000000 * (EndVerify.tv_sec - StartVerify.tv_sec) + EndVerify.tv_usec - StartVerify.tv_usec;
    cout << "Setup time: " << Setup / 1000 << " ms" << endl;
    cout << "Generate Proof time: " << Proof / 1000 << " ms" << endl;
    cout << "Verify time: " << Verify / 1000 << " ms" << endl;
    size_t proof_size = proof.size_in_bits();
    cout << "Proof size: " << proof_size << " bits" << endl;
    return 0;
}