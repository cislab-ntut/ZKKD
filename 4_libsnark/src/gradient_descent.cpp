#include <libsnark/common/default_types/r1cs_gg_ppzksnark_pp.hpp>
#include <libsnark/zk_proof_systems/ppzksnark/r1cs_gg_ppzksnark/r1cs_gg_ppzksnark.hpp>
#include <libsnark/gadgetlib1/pb_variable.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <nlohmann/json.hpp>
#include <sys/time.h>

using namespace libsnark;
using namespace std;
using json = nlohmann::json;

typedef libff::Fr<default_r1cs_gg_ppzksnark_pp> FieldT;

struct GradientDescent
{
    vector<int> param_original;
    vector<int> descent;
    vector<int> param_updated;
};

void to_json(json &j, const GradientDescent &gradientdescent)
{
    j = json{
        {"param_original", gradientdescent.param_original},
        {"descent", gradientdescent.descent},
        {"param_updated", gradientdescent.param_updated}};
}

void from_json(const json &j, GradientDescent &gradientdescent)
{
    gradientdescent.param_original = j.at("param_original").get<vector<int>>();
    gradientdescent.descent = j.at("descent").get<vector<int>>();
    gradientdescent.param_updated = j.at("param_updated").get<vector<int>>();
}

int main()
{
    ifstream file("extraction_gradient_descent/student_gradient_descent.json");

    if (!file)
    {
        cerr << "Unable to open file!" << endl;
        return 1;
    }

    default_r1cs_gg_ppzksnark_pp::init_public_params();
    protoboard<FieldT> pb;

    json inputs;
    file >> inputs;

    for (const auto &input : inputs)
    {
        GradientDescent gd = input.get<GradientDescent>();
        size_t size = gd.param_original.size();
        vector<pb_variable<FieldT>> param_original_vars(size);
        vector<pb_variable<FieldT>> descent_vars(size);
        vector<pb_variable<FieldT>> param_updated_vars(size);
        // gd.param_original - gd.descent = gd.param_updated

        for (size_t gd_index = 0; gd_index < size; gd_index++)
        {
            param_original_vars[gd_index].allocate(pb, "param_original_" + to_string(gd_index));
            descent_vars[gd_index].allocate(pb, "descent_" + to_string(gd_index));
            param_updated_vars[gd_index].allocate(pb, "param_updated_" + to_string(gd_index));

            pb.val(param_original_vars[gd_index]) = gd.param_original[gd_index];
            pb.val(descent_vars[gd_index]) = gd.descent[gd_index];
            pb.val(param_updated_vars[gd_index]) = gd.param_updated[gd_index];

            pb.add_r1cs_constraint(r1cs_constraint<FieldT>(param_original_vars[gd_index] - descent_vars[gd_index], 1, param_updated_vars[gd_index]));
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