#include "CUDASequenceComparatorV1.cuh"

__device__ float compare_sequences_kernel(char* miRNA_sequences_array, char* mRNA_sequences_array, int miRNA_start_index, int mRNA_start_index, int miRNA_len, int mRNA_len, int anneal_start_index) {
	int successful_matches = 0;
	int all_characters = miRNA_len;

	if (miRNA_len > mRNA_len) {
		printf("miRNA_len > mRNA_len, returning -1");
		return -1;
	}

	int mRNA_nucleotide_anneal_start = mRNA_start_index + anneal_start_index;
	for (int i = mRNA_nucleotide_anneal_start; i < (mRNA_nucleotide_anneal_start + miRNA_len); i++) {
		if (i > mRNA_start_index + mRNA_len) {
			break;
		}
		char current_miRNA_char = miRNA_sequences_array[i - mRNA_nucleotide_anneal_start + miRNA_start_index]; // start at miRNA_start_index, then go by 1 all the way until miRNA_len
		char current_mRNA_char = mRNA_sequences_array[i];
		successful_matches++;

		// TODO: IMPLEMENT match_strings_array as in CUDASequenceComparator/compare_sequences_kernel_se_log
	}
	return (float)successful_matches / all_characters;
}

__global__ void compare_sequence_arrays_kernel(float* d_result_array, char* d_miRNA_sequences_array, char* d_mRNA_sequences_array, char* d_mRNA_sequences_reversed_array, int miRNA_pitch, int mRNA_pitch, int* d_miRNA_lengths, int* d_mRNA_lengths) {
	// blockIdx.x determines miRNA
	// threadIdx.x determines mRNA
	// blockDim.x = count mRNAs (amount of threads in a single block)
	int miRNA_start_index = blockIdx.x * miRNA_pitch;
	int mRNA_start_index = threadIdx.x * mRNA_pitch;

	int miRNA_length = d_miRNA_lengths[blockIdx.x];
	int mRNA_length = d_mRNA_lengths[threadIdx.x];

	int result_index = blockIdx.x * blockDim.x + threadIdx.x;
	float max_match_strength = 0.0f;

	for (int i = 0; i < mRNA_length - miRNA_length; i++) {
		// seqop 1 and 2
		float match_strength_straight = compare_sequences_kernel(d_miRNA_sequences_array, d_mRNA_sequences_array, miRNA_start_index, mRNA_start_index, miRNA_length, mRNA_length, i);
		float match_strength_reversed = compare_sequences_kernel(d_miRNA_sequences_array, d_mRNA_sequences_array, miRNA_start_index, mRNA_start_index, miRNA_length, mRNA_length, i);

		float match_strength;
		// determine if there is a better match_strength for straight or reverse sequence - choose the one with greater strength
		if (match_strength_straight > match_strength_reversed) {
			match_strength = match_strength_straight;
		}
		else {
			match_strength = match_strength_reversed;
		}
		// assign max_match_strength the value of match_strength, if match_strength is greater
		if (match_strength > max_match_strength) {
			max_match_strength = match_strength;
		}
	}

	printf("Parallel thread [%d,%d] result at index %d: %f\n", blockIdx.x, threadIdx.x, result_index, max_match_strength);
	d_result_array[result_index] = max_match_strength;
}

CUDASequenceComparatorV1::CUDASequenceComparatorV1(std::string miRNA_filepath, std::string mRNA_filepath) {

	this->max_miRNA_length = 0;
	this->max_mRNA_length = 0;

	this->miRNAs_filepath = miRNA_filepath;
	std::vector<std::vector<std::string>> miRNAfile_process_result = process_miRNAsequences_file(miRNA_filepath);
	this->miRNA_ids = miRNAfile_process_result[0];
	this->miRNA_names = miRNAfile_process_result[1];
	this->miRNA_sequences = miRNAfile_process_result[2];
	
	this->mRNAs_filepath = mRNA_filepath;
	std::vector<std::vector<std::string>> mRNA_file_process_result = process_mRNAsequences_file(mRNA_filepath);
	this->mRNA_ids = mRNA_file_process_result[0];
	this->mRNA_sequences = mRNA_file_process_result[1];
	
	// create and populate int arrays of sequence lengths
	this->miRNA_lengths = new int[miRNA_sequences.size()];
	this->mRNA_lengths = new int[mRNA_sequences.size()];
	count_sequence_lengths(miRNA_lengths, miRNA_sequences);
	count_sequence_lengths(mRNA_lengths, mRNA_sequences);

	// allocate memory for char arrays
	this->miRNA_sequences_chars = (char*)malloc(this->miRNA_sequences.size() * this->max_miRNA_length * sizeof(char));
	this->mRNA_sequences_chars = (char*)malloc(this->mRNA_sequences.size() * this->max_mRNA_length * sizeof(char));
	this->mRNA_sequences_chars_reversed = (char*)malloc(this->mRNA_sequences.size() * this->max_mRNA_length * sizeof(char));
	memset(miRNA_sequences_chars, 0, sizeof(char) * this->miRNA_sequences.size() * this->max_miRNA_length);
	memset(mRNA_sequences_chars, 0, sizeof(char) * this->mRNA_sequences.size() * this->max_mRNA_length);
	memset(mRNA_sequences_chars_reversed, 0, sizeof(char) * this->mRNA_sequences.size() * this->max_mRNA_length);

	// populate this->miRNA_sequences_chars and this->mRNA_sequences_chars
	std::cout << "Processing miRNA_sequences_chars and mRNA_sequences_chars." << std::endl;
	StringUtils::convert_strings_to_Cstrings_ptr(miRNA_sequences_chars, miRNA_sequences, miRNA_sequences.size(), this->max_miRNA_length);
	StringUtils::convert_strings_to_Cstrings_ptr(mRNA_sequences_chars, mRNA_sequences, mRNA_sequences.size(), this->max_mRNA_length);
	StringUtils::reverse_array(mRNA_sequences_chars_reversed, mRNA_sequences_chars, mRNA_sequences.size(), mRNA_lengths, max_mRNA_length);

	this->sequence_comparison_results = (float*)malloc(miRNA_sequences.size() * mRNA_sequences.size() * sizeof(float));

	printf("Completed CUDASequenceComparatorV1 setup. Metadata:\n");
	printf("    - miRNA count: %d\n    - mRNA_count: %d\n", (int)miRNA_ids.size(), (int)mRNA_ids.size());
	printf("    - max miRNA len: %d\n    - max mRNA len: %d\n", (int)max_miRNA_length, (int)max_mRNA_length);
	
}

CUDASequenceComparatorV1::~CUDASequenceComparatorV1() {
	std::cout << "SequenceComparator destructor called" << std::endl;
}

/* 
 * Processes the miRNAsequences file, which is generated from the function util.save_mirbase_hsap_miRNAs from the python part of this project.
 * The file has the following structure:
 *
 *		AC (record.id), ID (record.name), SEQ (record.seq), base pair count
 *      MI0000060 \t hsa-let-7a-1 \t ugggaugagg uaguagguug uauaguuuua gggucacacc caccacuggg agauaacuau acaaucuacu gucuuuccua \t 80
 *
 * This function iterates through each line of the file and appends each of the first three elements (record.id, record.name, record.seq) in
 * a respective array (miRNA_miRDB_ids for record.id, miRNA_names for record.name, miRNA_sequences for record.seq). When the function is done
 * reading lines, it returns all these three arrays in a parent std::vector element.
 *
 * @param miRNA_sequences_hsa_only_filepath: The filepath to the miRNA sequences file generated by util.save_mirbase_hasp_miRNAs (in python code),
 * usually located at src_data_files/miRNAdbs/mirbase_miRNA_hsa-only.txt
 *
 * @return a std::vector containing three std::vector elements:
 *   - [0]: std::vector<std::string> miRNA_miRDB_ids
 *   - [1]: std::vector<std::string> miRNA_names
 *   - [2]: std::vector<std::string> miRNA_sequences
*/
std::vector<std::vector<std::string>> CUDASequenceComparatorV1::process_miRNAsequences_file(std::string miRNA_sequences_hsa_only_filepath) {
	std::vector<std::string> miRNA_miRDB_ids;
	std::vector<std::string> miRNA_names;
	std::vector<std::string> miRNA_sequences;

	std::ifstream file(miRNA_sequences_hsa_only_filepath);
	if (file.is_open()) {
		std::string line;
		while (std::getline(file, line)) {
			//std::vector<std::string> line_elements = StringUtils::split(line, "\t");
			if (line == "") {
				continue;
			}
			if (StringUtils::contains(line, "#")) {
				continue;
			}
			std::vector<std::string> line_elements;
			StringUtils::split(line, "\t", line_elements);
			miRNA_miRDB_ids.push_back(line_elements[0]);
			miRNA_names.push_back(line_elements[1]);
			miRNA_sequences.push_back(line_elements[2]);

			// process max miRNA length for diagnostic purposes
			if (line_elements[2].length() > this->max_miRNA_length) {
				this->max_miRNA_length = line_elements[2].length();
			}
		}
	}

	std::vector<std::vector<std::string>> result;
	result.push_back(miRNA_miRDB_ids);
	result.push_back(miRNA_names);
	result.push_back(miRNA_sequences);
	return result;
}

/*
 * Processes the mRNA sequences file (usually in constants(.py).TARGET_FOLDER/product_mRNAs_cpp.txt. In that file, mRNA elements are stored in the
 * following manner:
 *
 *     mRNA_uniprot_id \t mRNA_sequence \t mRNA_refseq_id1 \t mRNA_refseq_id2 \t [other mRNA_refseq_ids]
 *
 * This function processes each line and appends mRNA_uniprot_id, mRNA_sequence and refseq_ids into respective std::vector arrays. Since there can
 * be multiple refseq_ids per element, they are set directly into the member field (this->mRNA_refseq_ids) instead of being returned. The returned values
 * are only an std::vector<std::string> of mRNA uniprot ids and std::vector<std::string> of mRNA sequences.
 *
 * @param mRNA_sequences_filepath: The parameter to the product_mRNAs_cpp.txt generated at the end of dev_mrna_download.py or from a product_mRNA_refseq.json file
 * that is passed to the util.save_mRNAs_for_cpp (in util.py).
 *
 * @return:
 *   - [0]: std::vector<std::string> mRNA_uniprot_ids; a vector of uniprot ids
 *   - [1]: std::vector<std::string> mRNA_sequences; a vector of respective mRNA sequences
 *   Note: mRNA_refseq_ids are stored directly into this->mRNA_refseq_ids
 */
std::vector<std::vector<std::string>> CUDASequenceComparatorV1::process_mRNAsequences_file(std::string mRNA_sequences_filepath) {
	std::vector<std::string> mRNA_uniprot_ids;
	std::vector<std::string> mRNA_sequences;

	std::ifstream file(mRNA_sequences_filepath);
	if (file.is_open()) {
		std::string line;
		int line_index = 0;
		while (std::getline(file, line)) {
			if (line == "") {
				continue;
			}
			if (StringUtils::contains(line, "#")) {
				line_index++;
				continue;
			}
			std::vector<std::string> line_elements; StringUtils::split(line, "\t", line_elements);
			std::string mRNA_uniprot_id = line_elements[0];
			std::string mRNA_sequence = line_elements[1];
			if ((mRNA_uniprot_id == "None") || (mRNA_sequence == "None")) {
				// this can happen if during dev_mRNA_download.py ensembl doesn't find mRNA sequence and is normal: skip this line
				line_index++;
				continue;
			}
			// process max_mRNA_length for diagnostic purposes
			if (mRNA_sequence.length() > this->max_mRNA_length) {
				this->max_mRNA_length = mRNA_sequence.length();
			}
			// MAIN DEBUG LINE
			//std::cout << "Processing: " << line_index << "; uid = " << mRNA_uniprot_id << "; mRNA_length = " << mRNA_sequence.length() << std::endl;

			mRNA_uniprot_ids.push_back(mRNA_uniprot_id);
			mRNA_sequences.push_back(mRNA_sequence);

			// std::cout << "	line_elements.size() = " << line_elements.size() << std::endl;
			// std::cout << "	std::vector<std::string> mRNA_sequences sizeof " << sizeof(mRNA_sequences) << ", element count = " << mRNA_sequences.size() << std::endl;

			// directly load all of the other line_elements into this->mRNA_refseq_ids
			std::vector<std::string> current_mRNA_refseq_ids;
			for (int i = 0; i < line_elements.size(); i++) {
				if ((i == 0) || (i == 1)) {
					// don't load [0] and [1], since they are mRNA_uniprot_id and mRNA_sequence
					continue;
				}
				current_mRNA_refseq_ids.push_back(line_elements[i]);
			}
			this->mRNA_refseq_ids.push_back(current_mRNA_refseq_ids);
			line_index++;
		}
	}

	std::vector<std::vector<std::string>> result;
	result.push_back(mRNA_uniprot_ids);
	result.push_back(mRNA_sequences);
	return result;
}

/*
 * Loops over 'sequences' and populates the dst_array pointer with the lengts of the sequences. Note: when passing dst_array,
 * you need to precede the pointer variable to the array with the reference operator (&).
 */
void CUDASequenceComparatorV1::count_sequence_lengths(int*& dst_array, std::vector<std::string> sequences) {
	// todo: check that the sizes of dst_array and sequences match
	for (int i = 0; i < sequences.size(); i++) {
		dst_array[i] = sequences[i].length();
	}
}

void CUDASequenceComparatorV1::compare_sequences() {
	int num_blocks = miRNA_sequences.size();
	int num_threads = mRNA_sequences.size(); // num threads per each block

	float* d_result_array;
	char* d_miRNA_sequences_array;
	char* d_mRNA_sequences_array;
	char* d_mRNA_sequences_reversed_array;
	int* d_miRNA_lengths;
	int* d_mRNA_lengths;

	int miRNA_pitch = max_miRNA_length;
	int mRNA_pitch = max_mRNA_length;

	// set the primary GPU
	cudaSetDevice(0);

	// allocate GPU memory
	cudaMalloc((void**)&d_result_array, num_blocks * num_threads * sizeof(float));
	cudaMalloc((void**)&d_miRNA_sequences_array, miRNA_sequences.size() * miRNA_pitch * sizeof(char));
	cudaMalloc((void**)&d_mRNA_sequences_array, mRNA_sequences.size() * mRNA_pitch * sizeof(char));
	cudaMalloc((void**)&d_mRNA_sequences_reversed_array, mRNA_sequences.size() * mRNA_pitch * sizeof(char));
	cudaMalloc((void**)&d_miRNA_lengths, miRNA_sequences.size() * sizeof(int));
	cudaMalloc((void**)&d_mRNA_lengths, mRNA_sequences.size() * sizeof(int));

	// copy data from host to device
	cudaMemcpy(d_miRNA_sequences_array, this->miRNA_sequences_chars, miRNA_sequences.size() * miRNA_pitch * sizeof(char), cudaMemcpyHostToDevice);
	cudaMemcpy(d_mRNA_sequences_array, this->mRNA_sequences_chars, mRNA_sequences.size() * mRNA_pitch * sizeof(char), cudaMemcpyHostToDevice);
	cudaMemcpy(d_mRNA_sequences_reversed_array, this->mRNA_sequences_chars_reversed, mRNA_sequences.size() * mRNA_pitch * sizeof(char), cudaMemcpyHostToDevice);
	cudaMemcpy(d_miRNA_lengths, this->miRNA_lengths, miRNA_sequences.size() * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_mRNA_lengths, this->mRNA_lengths, mRNA_sequences.size() * sizeof(int), cudaMemcpyHostToDevice);

	// test copy back for debugging purposes - uncomment this block and put a breakpoint in the end to test if values are copied correctly
	/*
	char* h_miRNA_sequences_array_res = (char*)malloc(this->miRNA_sequences.size() * this->max_miRNA_length * sizeof(char));
	char* h_mRNA_sequences_array_res = (char*)malloc(this->mRNA_sequences.size() * this->max_mRNA_length * sizeof(char));
	char* h_mRNA_sequences_reversed_array_res = (char*)malloc(this->mRNA_sequences.size() * this->max_mRNA_length * sizeof(char));
	cudaMemcpy(h_miRNA_sequences_array_res, d_miRNA_sequences_array, miRNA_sequences.size() * miRNA_pitch * sizeof(char), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_mRNA_sequences_array_res, d_mRNA_sequences_array, mRNA_sequences.size() * mRNA_pitch * sizeof(char), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_mRNA_sequences_reversed_array_res, d_mRNA_sequences_reversed_array, mRNA_sequences.size() * mRNA_pitch * sizeof(char), cudaMemcpyDeviceToHost);
	std::vector<std::string> h_miRNA_seq_vec;
	std::vector<std::string> h_mRNA_seq_vec;
	std::vector<std::string> h_mRNA_seq_rev_vec;
	StringUtils::convert_Cstrings_to_strings_ptr(h_miRNA_seq_vec, h_miRNA_sequences_array_res, miRNA_sequences.size(), miRNA_pitch, ' ');
	StringUtils::convert_Cstrings_to_strings_ptr(h_mRNA_seq_vec, h_mRNA_sequences_array_res, mRNA_sequences.size(), mRNA_pitch, ' ');
	StringUtils::convert_Cstrings_to_strings_ptr(h_mRNA_seq_rev_vec, h_mRNA_sequences_reversed_array_res, mRNA_sequences.size(), mRNA_pitch, ' '); // set breakpoint here and check in debugger if values are copied correctly
	*/ 
	
	// run the gpu kernel
	compare_sequence_arrays_kernel <<<num_blocks, num_threads >>> (d_result_array, d_miRNA_sequences_array, d_mRNA_sequences_array, d_mRNA_sequences_reversed_array, miRNA_pitch, mRNA_pitch, d_miRNA_lengths, d_mRNA_lengths);
	cudaDeviceSynchronize();

	cudaMemcpy(this->sequence_comparison_results, d_result_array, num_blocks * num_threads * sizeof(float), cudaMemcpyDeviceToHost);

	// free device memory
	cudaFree(d_result_array);
	cudaFree(d_miRNA_sequences_array);
	cudaFree(d_mRNA_sequences_array);
	cudaFree(d_mRNA_sequences_reversed_array);
	cudaFree(d_miRNA_lengths);
	cudaFree(d_mRNA_lengths);
}