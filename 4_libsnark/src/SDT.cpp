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
vector<int> vec_input;
vector<int> vec_output;
static int SCALE_FACTOR = 1024; // 2^10

uint64_t ROOT_ID;

struct Node
{
    uint64_t node_id;
    bool is_leaf;
    bool is_root;
    uint64_t left_child_id;
    uint64_t right_child_id;
    vector<int> input;
    vector<int> fc_weight;
    int fc_bias;
    int linear_output;
    int beta;
    int beta_output;
    int sigmoid_w;
    int sigmoid_b;
    int prob;
    int path_prob_in;
    int path_prob_out_left;
    int path_prob_out_right;
};

struct Leaf
{
    uint64_t node_id;
    bool is_leaf;
    bool is_final_output;
    int path_prob;
    vector<int> Q;
};

void to_json(json &j, const Node &node)
{
    j = json{
        {"node_id", node.node_id},
        {"is_leaf", node.is_leaf},
        {"is_root", node.is_root},
        {"left_child_id", node.left_child_id},
        {"right_child_id", node.right_child_id},
        {"input", node.input},
        {"fc_weight", node.fc_weight},
        {"fc_bias", node.fc_bias},
        {"linear_output", node.linear_output},
        {"beta", node.beta},
        {"beta_output", node.beta_output},
        {"sigmoid_w", node.sigmoid_w},
        {"sigmoid_b", node.sigmoid_b},
        {"prob", node.prob},
        {"path_prob_in", node.path_prob_in},
        {"path_prob_out_left", node.path_prob_out_left},
        {"path_prob_out_right", node.path_prob_out_right}};
}

void from_json(const json &j, Node &node)
{
    node.node_id = j.at("node_id").get<uint64_t>();
    node.is_leaf = j.at("is_leaf").get<bool>();
    node.is_root = j.at("is_root").get<bool>();
    node.left_child_id = j.at("left_child_id").get<uint64_t>();
    node.right_child_id = j.at("right_child_id").get<uint64_t>();
    node.input = j.at("input").get<vector<int>>();
    node.fc_weight = j.at("fc_weight").get<vector<int>>();
    node.fc_bias = j.at("fc_bias").get<int>();
    node.linear_output = j.at("linear_output").get<int>();
    node.beta = j.at("beta").get<int>();
    node.beta_output = j.at("beta_output").get<int>();
    node.sigmoid_w = j.at("sigmoid_w").get<int>();
    node.sigmoid_b = j.at("sigmoid_b").get<int>();
    node.prob = j.at("prob").get<int>();
    node.path_prob_in = j.at("path_prob_in").get<int>();
    node.path_prob_out_left = j.at("path_prob_out_left").get<int>();
    node.path_prob_out_right = j.at("path_prob_out_right").get<int>();
}

void to_json(json &j, const Leaf &leaf)
{
    j = json{
        {"node_id", leaf.node_id},
        {"is_leaf", leaf.is_leaf},
        {"is_final_output", leaf.is_final_output},
        {"path_prob", leaf.path_prob},
        {"Q", leaf.Q}};
}

void from_json(const json &j, Leaf &leaf)
{
    leaf.node_id = j.at("node_id").get<uint64_t>();
    leaf.is_leaf = j.at("is_leaf").get<bool>();
    leaf.is_final_output = j.at("is_final_output").get<bool>();
    leaf.path_prob = j.at("path_prob").get<int>();
    leaf.Q = j.at("Q").get<vector<int>>();
}

int python_mod(int a, int b)
{
    int r = a % b;
    return r < 0 ? r + abs(b) : r;
}

int main()
{
    ifstream file("extraction_student/student_extraction.json");

    if (!file)
    {
        cerr << "Unable to open file!" << endl;
        return 1;
    }

    // 讀取 JSON
    json nodes;
    file >> nodes;

    unordered_map<uint64_t, Node> nodes_map;
    unordered_map<uint64_t, Leaf> leaves_map;
    for (const auto &input : nodes)
    {
        if (input["is_leaf"].get<bool>())
        {
            if (input["is_final_output"].get<bool>())
            {
                vec_output = input["Q"].get<vector<int>>();
            }
            Leaf leaf;
            from_json(input, leaf);
            leaves_map[leaf.node_id] = leaf;
        }
        else
        {
            if (input["is_root"].get<bool>())
            {
                ROOT_ID = input["node_id"].get<uint64_t>();
            }
            Node node;
            from_json(input, node);
            nodes_map[node.node_id] = node;
        }
    }

    // 提出 root 的input
    if (nodes_map.find(ROOT_ID) != nodes_map.end())
    {
        const Node &root = nodes_map[ROOT_ID];
        vec_input = root.input;
    }

    // initialize the curve parameters
    default_r1cs_gg_ppzksnark_pp::init_public_params();
    protoboard<FieldT> pb;

    // public input, output
    pb_variable<FieldT> input[784];
    pb_variable<FieldT> final_output[10];

    // nodes inner calculation
    // pb_variable<FieldT> fc_weight[784];
    // pb_variable<FieldT> fc_bias;
    // pb_variable<FieldT> linear_output;
    // pb_variable<FieldT> beta;
    // pb_variable<FieldT> beta_output;
    // pb_variable<FieldT> sigmoid_w;
    // pb_variable<FieldT> sigmoid_b;
    // pb_variable<FieldT> prob;

    // Set input
    for (int input_index = 0; input_index < 784; input_index++)
    {
        string name = "input_" + to_string(input_index);
        input[input_index].allocate(pb, name);
        pb.val(input[input_index]) = vec_input[input_index];
    }

    // Set output
    for (int output_index = 0; output_index < 10; output_index++)
    {
        string name = "output_" + to_string(output_index);
        final_output[output_index].allocate(pb, name);
        pb.val(final_output[output_index]) = vec_output[output_index];
    }

    // Set SCALE_FACTOR
    pb_variable<FieldT> scale_factor;
    scale_factor.allocate(pb, "scale_factor");
    pb.val(scale_factor) = SCALE_FACTOR;

    queue<uint64_t> inner_node_queue, leaves_queue;
    // 建立所有 inner node 的 queue
    for (const auto &node_pair : nodes_map)
    {
        uint64_t node_id = node_pair.first;
        // 內部節點
        inner_node_queue.push(node_id);
    }

    for (const auto &leaf_pair : leaves_map)
    {
        uint64_t leaf_id = leaf_pair.first;
        // 葉節點
        leaves_queue.push(leaf_id);
    }

    // Set inner node calculation
    while (!inner_node_queue.empty())
    {
        uint64_t query_id = inner_node_queue.front();
        inner_node_queue.pop();

        const Node &node = nodes_map[query_id];
        pb_variable<FieldT> fc_weight[784];
        // pb_variable<FieldT> fc_weight_mul_temp[784];
        // pb_variable<FieldT> fc_weight_add_temp[783];
        pb_variable<FieldT> fc_bias;
        pb_variable<FieldT> linear_output;
        pb_variable<FieldT> beta;
        pb_variable<FieldT> beta_output;
        pb_variable<FieldT> sigmoid_w;
        pb_variable<FieldT> sigmoid_b;
        pb_variable<FieldT> prob;

        // Set fc_weight
        for (int fc_weight_index = 0; fc_weight_index < 784; fc_weight_index++)
        {
            string name = to_string(query_id) + "_fc_weight_" + to_string(fc_weight_index);
            fc_weight[fc_weight_index].allocate(pb, name);
            pb.val(fc_weight[fc_weight_index]) = node.fc_weight[fc_weight_index];
        }

        // Set fc_bias
        string fc_bias_name = to_string(query_id) + "_fc_bias";
        fc_bias.allocate(pb, fc_bias_name);
        pb.val(fc_bias) = node.fc_bias;

        // Set linear_output
        string linear_output_name = to_string(query_id) + "_linear_output";
        linear_output.allocate(pb, linear_output_name);
        pb.val(linear_output) = node.linear_output;

        // Set beta
        string beta_name = to_string(query_id) + "_beta";
        beta.allocate(pb, beta_name);
        pb.val(beta) = node.beta;

        // Set beta_output
        string beta_output_name = to_string(query_id) + "_beta_output";
        beta_output.allocate(pb, beta_output_name);
        pb.val(beta_output) = node.beta_output;

        // Set sigmoid_w
        string sigmoid_w_name = to_string(query_id) + "_sigmoid_w";
        sigmoid_w.allocate(pb, sigmoid_w_name);
        pb.val(sigmoid_w) = node.sigmoid_w;

        // Set sigmoid_b
        string sigmoid_b_name = to_string(query_id) + "_sigmoid_b";
        sigmoid_b.allocate(pb, sigmoid_b_name);
        pb.val(sigmoid_b) = node.sigmoid_b;

        // Set prob
        string prob_name = to_string(query_id) + "_prob";
        prob.allocate(pb, prob_name);
        pb.val(prob) = node.prob;

        // set path_prob_in
        pb_variable<FieldT> path_prob_in;
        string path_prob_in_name = to_string(query_id) + "_path_prob_in";
        path_prob_in.allocate(pb, path_prob_in_name);
        pb.val(path_prob_in) = node.path_prob_in;

        // set path_prob_out_left
        pb_variable<FieldT> path_prob_out_left;
        string path_prob_out_left_name = to_string(query_id) + "_path_prob_out_left";
        path_prob_out_left.allocate(pb, path_prob_out_left_name);
        pb.val(path_prob_out_left) = node.path_prob_out_left;

        // set path_prob_out_right
        pb_variable<FieldT> path_prob_out_right;
        string path_prob_out_right_name = to_string(query_id) + "_path_prob_out_right";
        path_prob_out_right.allocate(pb, path_prob_out_right_name);
        pb.val(path_prob_out_right) = node.path_prob_out_right;

        // temp X * W for each element (in matrix)
        int arr_fc_weight_mul_temp[784];
        pb_variable<FieldT> fc_weight_mul_temp[784];
        for (int mul_index = 0; mul_index < 784; mul_index++)
        {
            arr_fc_weight_mul_temp[mul_index] = node.fc_weight[mul_index] * vec_input[mul_index];

            string name = to_string(query_id) + "_fc_weight_mul_temp_" + to_string(mul_index);
            fc_weight_mul_temp[mul_index].allocate(pb, name);
            pb.val(fc_weight_mul_temp[mul_index]) = arr_fc_weight_mul_temp[mul_index];
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

            string name = to_string(query_id) + "_fc_weight_add_temp_" + to_string(fc_weight_index);
            fc_weight_add_temp[fc_weight_index].allocate(pb, name);
            pb.val(fc_weight_add_temp[fc_weight_index]) = arr_fc_weight_add_temp[fc_weight_index];
        }

        // Add R1CS constraints
        // X * W
        for (int fc_weight_index = 0; fc_weight_index < 784; fc_weight_index++)
        {
            pb.add_r1cs_constraint(r1cs_constraint<FieldT>(input[fc_weight_index], fc_weight[fc_weight_index], fc_weight_mul_temp[fc_weight_index]));
        }

        // sum ([X * W])
        for (int fc_weight_index = 0; fc_weight_index < 783; fc_weight_index++)
        {
            if (fc_weight_index == 0)
            {
                pb.add_r1cs_constraint(r1cs_constraint<FieldT>(fc_weight_mul_temp[fc_weight_index] + fc_weight_mul_temp[fc_weight_index + 1], 1, fc_weight_add_temp[fc_weight_index]));
            }
            else
            {
                pb.add_r1cs_constraint(r1cs_constraint<FieldT>(fc_weight_add_temp[fc_weight_index - 1] + fc_weight_mul_temp[fc_weight_index + 1], 1, fc_weight_add_temp[fc_weight_index]));
            }
        }

        // sum ([X * W]) rescale
        int int_fc_weight_add_temp_rescale_mod = python_mod(arr_fc_weight_add_temp[782], SCALE_FACTOR);
        // cout << "-3759824 % 1024 = " << python_mod(-3759824,1024) << "," << int_fc_weight_add_temp_rescale_mod << endl;
        // cout << arr_fc_weight_add_temp[782] << endl;
        // cout << int_fc_weight_add_temp_rescale_mod << endl;
        // cout << node.fc_bias << endl;
        // cout << node.linear_output << endl;

        pb_variable<FieldT> fc_weight_add_temp_rescale_mod;
        string fc_weight_add_temp_rescale_name = to_string(query_id) + "_fc_weight_add_temp_rescale_mod";
        fc_weight_add_temp_rescale_mod.allocate(pb, fc_weight_add_temp_rescale_name);
        pb.val(fc_weight_add_temp_rescale_mod) = int_fc_weight_add_temp_rescale_mod;
        // (fc_weight_add_temp[782] - arr_fc_weight_add_temp[782] % SCALE_FACTOR) / SCALE_FACTOR + fc_bias = linear_output
        // -> fc_weight_add_temp[782] - arr_fc_weight_add_temp[782] % SCALE_FACTOR = SCALE_FACTOR * (linear_output - fc_bias)
        pb.add_r1cs_constraint(r1cs_constraint<FieldT>(linear_output - fc_bias, scale_factor, fc_weight_add_temp[782] - fc_weight_add_temp_rescale_mod));

        // beta * linear_output / 1024 = beta_output ... beta * linear_output % 1024
        // (beta * linear_output) - (beta * linear_output) % 2^10 = beta_output * 2^10

        int int_beta_output_temp = node.beta * node.linear_output;
        pb_variable<FieldT> beta_output_temp;
        string beta_output_temp_name = to_string(query_id) + "_beta_output_temp";
        beta_output_temp.allocate(pb, beta_output_temp_name);
        pb.val(beta_output_temp) = int_beta_output_temp;

        int int_beta_linear_output_mod = python_mod(int_beta_output_temp, SCALE_FACTOR);
        pb_variable<FieldT> beta_linear_output_mod;
        string beta_linear_output_mod_name = to_string(query_id) + "_beta_linear_output_mod";
        beta_linear_output_mod.allocate(pb, beta_linear_output_mod_name);
        pb.val(beta_linear_output_mod) = int_beta_linear_output_mod;

        pb.add_r1cs_constraint(r1cs_constraint<FieldT>(beta, linear_output, beta_output_temp));
        pb.add_r1cs_constraint(r1cs_constraint<FieldT>(beta_output, scale_factor, beta_output_temp - beta_linear_output_mod));

        // sigmoid: (beta_output * sigmoid_w + sigmoid_b) / 1024 = prob ... (beta_output * sigmoid_w + sigmoid_b) % 1024
        // (beta_output * sigmoid_w + sigmoid_b) - (beta_output * sigmoid_w + sigmoid_b) % 2^10 = prob * 2^10
        int int_sigmoid_output_temp = node.beta_output * node.sigmoid_w + node.sigmoid_b;
        pb_variable<FieldT> sigmoid_output_temp;
        string sigmoid_output_temp_name = to_string(query_id) + "_sigmoid_output_temp";
        sigmoid_output_temp.allocate(pb, sigmoid_output_temp_name);
        pb.val(sigmoid_output_temp) = int_sigmoid_output_temp;

        int int_sigmoid_output_mod = python_mod(int_sigmoid_output_temp, SCALE_FACTOR);
        pb_variable<FieldT> sigmoid_output_mod;
        string sigmoid_output_mod_name = to_string(query_id) + "_sigmoid_output_mod";
        sigmoid_output_mod.allocate(pb, sigmoid_output_mod_name);
        pb.val(sigmoid_output_mod) = int_sigmoid_output_mod;

        pb.add_r1cs_constraint(r1cs_constraint<FieldT>(beta_output, sigmoid_w, sigmoid_output_temp - sigmoid_b));
        pb.add_r1cs_constraint(r1cs_constraint<FieldT>(prob, scale_factor, sigmoid_output_temp - sigmoid_output_mod));

        // descale(path_prob_in * prob) = path_prob_out_right
        // 2^10 - path_prob_out_right = path_prob_out_left
        int temp_path_prob_out_right = node.path_prob_in * node.prob;
        pb_variable<FieldT> path_prob_out_right_temp;
        string path_prob_out_right_temp_name = to_string(query_id) + "_path_prob_out_right_temp";
        path_prob_out_right_temp.allocate(pb, path_prob_out_right_temp_name);
        pb.val(path_prob_out_right_temp) = temp_path_prob_out_right;

        int temp_path_prob_out_right_mod = python_mod(temp_path_prob_out_right, SCALE_FACTOR);
        pb_variable<FieldT> path_prob_out_right_mod;
        string path_prob_out_right_mod_name = to_string(query_id) + "_path_prob_out_right_mod";
        path_prob_out_right_mod.allocate(pb, path_prob_out_right_mod_name);
        pb.val(path_prob_out_right_mod) = temp_path_prob_out_right_mod;

        pb.add_r1cs_constraint(r1cs_constraint<FieldT>(prob, path_prob_in, path_prob_out_right_temp));
        // path_prob_out_right_temp // 1024 = path_prob_out_right ... path_prob_out_right_temp % 1024
        // path_prob_out_right * scale_factor  = path_prob_out_right_temp - path_prob_out_right_temp % 1024
        pb.add_r1cs_constraint(r1cs_constraint<FieldT>(path_prob_out_right, scale_factor, path_prob_out_right_temp - path_prob_out_right_mod));

        int temp_path_prob_out_left = node.path_prob_in * (SCALE_FACTOR - node.prob);
        pb_variable<FieldT> path_prob_out_left_temp;
        string path_prob_out_left_temp_name = to_string(query_id) + "_path_prob_out_left_temp";
        path_prob_out_left_temp.allocate(pb, path_prob_out_left_temp_name);
        pb.val(path_prob_out_left_temp) = temp_path_prob_out_left;

        int temp_path_prob_out_left_mod = python_mod(temp_path_prob_out_left, SCALE_FACTOR);
        pb_variable<FieldT> path_prob_out_left_mod;
        string path_prob_out_left_mod_name = to_string(query_id) + "_path_prob_out_left_mod";
        path_prob_out_left_mod.allocate(pb, path_prob_out_left_mod_name);
        pb.val(path_prob_out_left_mod) = temp_path_prob_out_left_mod;

        pb.add_r1cs_constraint(r1cs_constraint<FieldT>(scale_factor - prob, path_prob_in, path_prob_out_left_temp));
        pb.add_r1cs_constraint(r1cs_constraint<FieldT>(path_prob_out_left, scale_factor, path_prob_out_left_temp - path_prob_out_left_mod));
    }

    pb.set_input_sizes(794);

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