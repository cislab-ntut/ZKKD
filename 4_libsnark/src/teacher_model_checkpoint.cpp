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
    int choose_neuron;
    vector<int> input;
    vector<int> output;
    vector<int> checkpoint_weight;
    int checkpoint_bias;
    int calculated_output;
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
        {"choose_neuron", fc.choose_neuron},
        {"input", fc.input},
        {"output", fc.output},
        {"checkpoint_weight", fc.checkpoint_weight},
        {"checkpoint_bias", fc.checkpoint_bias},
        {"calculated_output", fc.calculated_output}};
}

void from_json(const json &j, FC &fc)
{
    fc.layer = j.at("layer").get<string>();
    fc.choose_neuron = j.at("choose_neuron").get<int>();
    fc.input = j.at("input").get<vector<int>>();
    fc.output = j.at("output").get<vector<int>>();
    fc.checkpoint_weight = j.at("checkpoint_weight").get<vector<int>>();
    fc.checkpoint_bias = j.at("checkpoint_bias").get<int>();
    fc.calculated_output = j.at("calculated_output").get<int>();
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
            "extraction_teacher/fc1/fc1_operation_log.json",
            "extraction_teacher/fc2/fc2_operation_log.json",
            "extraction_teacher/fc3/fc3_operation_log.json"};

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
            "extraction_teacher/relu1/relu1_operation_log.json",
            "extraction_teacher/relu2/relu2_operation_log.json"};

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

    // initialize the curve parameters
    default_r1cs_gg_ppzksnark_pp::init_public_params();
    protoboard<FieldT> pb;

    ReadFCParams fc_read;
    fc_read.read_operation_logs();

    for (size_t fc_read_index = 0; fc_read_index < fc_read.operation_logs.size(); fc_read_index++)
    {

        const FC fc_current = fc_read.operation_logs[fc_read_index];

        pb_variable<FieldT> input[784];
        pb_variable<FieldT> fc_checkpoint_weight[784];
        pb_variable<FieldT> fc_checkpoint_bias;
        pb_variable<FieldT> fc_calculated_output;

        // set input
        for (int input_index = 0; input_index < 784; input_index++)
        {
            string name = "input_" + to_string(input_index) + fc_current.layer;
            input[input_index].allocate(pb, name);
            pb.val(input[input_index]) = fc_current.input[input_index];
        }

        // set checkpoint weight
        for (int weight_index = 0; weight_index < 784; weight_index++)
        {
            string name = "weight_" + to_string(weight_index) + fc_current.layer;
            fc_checkpoint_weight[weight_index].allocate(pb, name);
            pb.val(fc_checkpoint_weight[weight_index]) = fc_current.checkpoint_weight[weight_index];
        }

        // set checkpoint bias
        string bias_name = "bias_" + fc_current.layer;
        fc_checkpoint_bias.allocate(pb, bias_name);
        pb.val(fc_checkpoint_bias) = fc_current.checkpoint_bias;

        // set calculated output
        string output_name = "output_" + fc_current.layer;
        fc_calculated_output.allocate(pb, output_name);
        pb.val(fc_calculated_output) = fc_current.calculated_output;

        // Set SCALE_FACTOR
        pb_variable<FieldT> scale_factor;
        scale_factor.allocate(pb, "scale_factor");
        pb.val(scale_factor) = SCALE_FACTOR;

        // calculate X * W
        int arr_fc_weight_mul_temp[784];
        pb_variable<FieldT> fc_weight_mul_temp[784];
        for (int mul_index = 0; mul_index < 784; mul_index++)
        {
            arr_fc_weight_mul_temp[mul_index] = fc_current.input[mul_index] * fc_current.checkpoint_weight[mul_index];
            string name = "fc_weight_mul_temp_" + to_string(mul_index) + fc_current.layer;
            fc_weight_mul_temp[mul_index].allocate(pb, name);
            pb.val(fc_weight_mul_temp[mul_index]) = arr_fc_weight_mul_temp[mul_index];
            pb.add_r1cs_constraint(r1cs_constraint<FieldT>(input[mul_index], fc_checkpoint_weight[mul_index], fc_weight_mul_temp[mul_index]));
        }

        // temp sum ([X * W])
        int arr_fc_weight_add_temp[783];
        pb_variable<FieldT> fc_weight_add_temp[783];
        for (int fc_weight_index = 0; fc_weight_index < 783; fc_weight_index++)
        {
            if (fc_weight_index == 0)
            {
                arr_fc_weight_add_temp[fc_weight_index] = arr_fc_weight_mul_temp[fc_weight_index] + arr_fc_weight_mul_temp[fc_weight_index + 1];
            }
            else
            {
                arr_fc_weight_add_temp[fc_weight_index] = arr_fc_weight_add_temp[fc_weight_index - 1] + arr_fc_weight_mul_temp[fc_weight_index + 1];
            }

            string name = "fc_weight_add_temp_" + to_string(fc_weight_index) + fc_current.layer;
            fc_weight_add_temp[fc_weight_index].allocate(pb, name);
            pb.val(fc_weight_add_temp[fc_weight_index]) = arr_fc_weight_add_temp[fc_weight_index];
        }

        // sum ([X * W]) rescale
        int int_fc_weight_add_temp_rescale_mod = python_mod(arr_fc_weight_add_temp[782], SCALE_FACTOR);
        pb_variable<FieldT> fc_weight_add_temp_rescale_mod;
        string fc_weight_add_temp_rescale_name = "fc_weight_add_temp_rescale_mod" + fc_current.layer;
        fc_weight_add_temp_rescale_mod.allocate(pb, fc_weight_add_temp_rescale_name);
        pb.val(fc_weight_add_temp_rescale_mod) = int_fc_weight_add_temp_rescale_mod;
        pb.add_r1cs_constraint(r1cs_constraint<FieldT>(fc_calculated_output - fc_checkpoint_bias, scale_factor, fc_weight_add_temp[782] - fc_weight_add_temp_rescale_mod));
    }

    ReadReLUParams relu_read;
    relu_read.read_operation_logs();
    for (size_t i = 0; i < relu_read.operation_logs.size(); i++)
    {
        ReLU current_relu = relu_read.operation_logs[i];
        pb_variable<FieldT> input[1200];
        pb_variable<FieldT> relu_op[1200];
        pb_variable<FieldT> output[1200];

        // set input
        for (int input_index = 0; input_index < 1200; input_index++)
        {
            string name = "input_" + to_string(input_index) + current_relu.layer;
            input[input_index].allocate(pb, name);
            pb.val(input[input_index]) = current_relu.input[input_index];
        }

        // set relu operator
        for (int relu_index = 0; relu_index < 1200; relu_index++)
        {
            string name = "relu_" + to_string(relu_index) + current_relu.layer;
            relu_op[relu_index].allocate(pb, name);
            pb.val(relu_op[relu_index]) = current_relu.relu_operator[relu_index];
        }

        // set output
        for (int output_index = 0; output_index < 1200; output_index++)
        {
            string name = "output_" + to_string(output_index) + current_relu.layer;
            output[output_index].allocate(pb, name);
            pb.val(output[output_index]) = current_relu.output[output_index];
        }

        for (int relu_index = 0; relu_index < 1200; relu_index++)
        {
            pb.add_r1cs_constraint(r1cs_constraint<FieldT>(input[relu_index], relu_op[relu_index], output[relu_index]));
        }
    }

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