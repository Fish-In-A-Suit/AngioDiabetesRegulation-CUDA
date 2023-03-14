#include "CUDASequenceComparator.cuh"

/*
 * Compares two sequences (anneals a miRNA onto a mRNA from anneal_start_index forwards) and returns the match_strength (no_matches / size of miRNA). Can be reused across
 * multiple kernels.
 */
__device__ float compare_sequences_kernel(char* miRNA_sequence, char* mRNA_sequence, int miRNA_len, int mRNA_len, int anneal_start_index) {
	int successful_matches = 0;
	int all_characters = miRNA_len;

	// miRNA_len should be less than mRNA_len
	if (miRNA_len > mRNA_len) {
		printf("miRNA_len > mRNA_len, returning -1");
		return -1;
	}

	for (int i = anneal_start_index; i < (anneal_start_index + miRNA_len); i++) {
		// safety check that it doesn't overvlow bounds, logic to not overflow SHOULD BE IMPLEMENTED OUTSIDE OF THIS KERNEL
		if (i > mRNA_len) {
			break;
		}
		char current_miRNA_char = miRNA_sequence[i];
		char current_mRNA_char = mRNA_sequence[i];
		if (current_miRNA_char == current_mRNA_char) {
			successful_matches++;
		}
	}
	// printf("  - successful_matches = %f, all_characters = %f", (float)successful_matches, (float)all_characters); // error when casting here !!!

	return (float)successful_matches / all_characters;
}

/*
 * A device (GPU) implementation of the std::strlen function to find the number of characters inside a const char* cstr.
 */
__device__ int my_strlen_dev(char* cstr) {
	int len = 0;
	while (cstr[len] != '\0') {
		len++;
	}
	return len;
}

/*
 * Prints out first 'char_count' and last 'char_count' nucleotides of a sequence.
 * 
 * You use "%.*s" with specific printf parameters to print out the substring of a char* array.
 * 
 *     printf("%.*s", AMOUNT_OF_CHARS_TO_PRINT, START_INDEX);
 * 
 * Since char* sequence is a pointer to the first character in the array, you can use offsets to specify the starting character,
 * eg. sequence + 7 would point to the 8th character in the char* array. 
 * 
 * Code example:
 *     char* str = "Hello world!";
 *     int start = 7;
 *     int char_count = 5;
 * 
 *     printf("%.*s\n", char_count, str + start); 
 *     // --> output: world
 * 
 * TODO: THIS FUNCTION IS SOMEHOW BUGGED. DEBUG IT.
 */
__device__ void debug_print_sequence_v1(char* sequence, int char_count) {
	int start_char_index = 0;
	int end_char_index = my_strlen_dev(sequence);

	printf("%.*s", char_count, sequence + start_char_index); // print first char_count elements
	printf(" ... ");
	printf(".*s\n", char_count, sequence + end_char_index - char_count); // print last char_count elements
}

__device__ void debug_print_sequence_v2(char* sequence, int char_count, int seq_length) {
	for (int i = 0; i < char_count; i++) {
		printf("%s", sequence[i]);
	}
	printf(" ... ");
	for (int i = seq_length - char_count; i < seq_length; i++) {
		printf("%s", sequence[i]);
	}
	printf("\n");
}

/*
 * Reverses the input char array and returns a pointer to the reversed char array
 * THIS PRODUCES ERRORS!!! DO NOT USE THIS!!!
 */
__device__ char* reverse_char_array(char* char_array_to_flip, int array_len) {
	// this is faulty, you cannot dynamically allocate like in C++
	// char* result_array = new char[array_len];

	// CUDA supporty dynamic shared memory allocation - see: https://stackoverflow.com/questions/5531247/allocating-shared-memory%5B/url%5D
	extern __shared__ char result_array[];
	
	for (int i = 0; i < array_len; i++) {
		char current_char = char_array_to_flip[array_len - 1 - i]; // -1 because array_len is 1 greater than what we access in loops
		result_array[i] = current_char;
	}
	return result_array;
}

/*
 * This function compares each miRNA against all mRNAs and stores best matches into d_result_array. Compare logic overview:
 * 
 * for each miRNA:
 *		for each mRNA:
 *			float max_match_strength = 0.0f
 *			for (int i = 0; i < (current_mRNA_length - current_miRNA_length); i++):
 *				match_strength = compare_sequences(...); // compare_sequences(char* miRNA_seq, char* mRNA_seq, int miRNA_len, int mRNA_len, int anneal_start_index)
 *				if match_strength > max_match_strength: max_match_strength = match_strength
 * 
 * There should be a number of blocks equal to the number of miRNAs. Each block should process one miRNA against all mRNAs. A block uses one thread to process one mRNA against a miRNA. Amount of threads per block = amount of mRNAs.
 * Call like this: compare_sequences<<<miRNAs.size(), mRNAs.size()>>>
 *			
 * 
 */
__global__ void compare_sequence_arrays_kernel(float* d_result_array, char(*d_miRNA_sequences)[Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH], char(*d_mRNA_sequences)[Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH], char(*d_mRNA_sequences_reversed)[Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH], int* d_miRNA_lengths, int* d_mRNA_lengths) {
	// printf("Running parallel thread [%d,%d]\n", blockIdx.x, threadIdx.x);
	// --- *** JUST A TEST *** ---
	// char current_miRNA[][Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH] = *(*(d_miRNA_sequences + 0) + 0); // not allowed
	char firstChar_miRNA = *(*(d_miRNA_sequences + 0) + 0);
	char* firstRow_miRNA = d_miRNA_sequences[0];
	char firstCharv1_miRNA = firstRow_miRNA[0];

	char firstChar_mRNA = *(*(d_mRNA_sequences + 0) + 0);
	char* firstRow_mRNA = d_mRNA_sequences[0];
	char firstCharv1_mRNA = firstRow_mRNA[0];
	// --- *** end of test *** ---

	// --- *** MAIN LOGIC *** ---
	char* current_miRNA = d_miRNA_sequences[blockIdx.x];
	char* current_mRNA = d_mRNA_sequences[threadIdx.x];
	char* current_mRNA_reversed = d_mRNA_sequences[threadIdx.x];
	int current_miRNA_length = d_miRNA_lengths[blockIdx.x];
	int current_mRNA_length = d_mRNA_lengths[threadIdx.x];
	int result_index = blockIdx.x * blockDim.x + threadIdx.x; // blockDim.x represents the number of threads in a block (= the number of mRNAs)

	// use this with nsight debugger to step over first 5 and last 5 characters!!!
	// printf("mRNA_sequence = ");
	// debug_print_sequence_v2(current_mRNA, 5, current_mRNA_length);
	// printf("mRNA_sequence_reversed = ");
	// debug_print_sequence_v2(current_mRNA_reversed, 5, current_mRNA_length);

	// char* current_miRNA_reversed = reverse_char_array(current_miRNA, current_miRNA_length);

	float max_match_strength = 0.0f;

	// anneal miRNA onto all mRNA substrings
	for (int i = 0; i < (current_mRNA_length - current_miRNA_length); i++) {
		// compare both the "straight" and the reversed miRNA sequences
		float match_strength_straight = compare_sequences_kernel(current_miRNA, current_mRNA, current_miRNA_length, current_mRNA_length, i);
		float match_strength_reversed = compare_sequences_kernel(current_miRNA, current_mRNA_reversed, current_miRNA_length, current_mRNA_length, i);
		//printf("    - %f, %f", match_strength_straight, match_strength_reversed);
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
	printf("Parallel thread [%d,%d] result: %f\n", blockIdx.x, threadIdx.x, max_match_strength);
	d_result_array[result_index] = max_match_strength;
}

CUDASequenceComparator::CUDASequenceComparator(std::string miRNAsequences_filepath, std::string mRNAsequences_filepath)
{
	this->max_miRNA_length = 0;
	this->max_mRNA_length = 0;
	
	this->miRNAs_filepath = miRNAsequences_filepath;
	std::vector<std::vector<std::string>> miRNAfile_process_result = process_miRNAsequences_file(miRNAsequences_filepath);
	this->miRNA_miRDB_ids = miRNAfile_process_result[0];
	this->miRNA_names = miRNAfile_process_result[1];
	this->miRNA_sequences = miRNAfile_process_result[2];
	std::cout << "miRNA_miRDB_ids, miRNA_names, miRNA_sequences sizes:  " << miRNA_miRDB_ids.size() << ", " << miRNA_names.size() << ", " << miRNA_sequences.size() << std::endl;

	this->mRNAs_filepath = mRNAsequences_filepath;
	std::vector<std::vector<std::string>> mRNA_file_process_result = process_mRNAsequences_file(mRNAsequences_filepath);
	this->mRNA_ids = mRNA_file_process_result[0];
	this->mRNA_sequences = mRNA_file_process_result[1];
	std::cout << "Processed counts for mRNA_ids, mRNA_sequences, mRNA_refseq_ids: " << mRNA_ids.size() << ", " << mRNA_sequences.size() << ", " << mRNA_refseq_ids.size() << std::endl;

	// char miRNA_cstrings_array[miRNA_sequences.size()][Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH]; // not allowed, as miRNA_sequences.size() is not const
	char(*miRNA_cstrings_array_ptr)[Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH]  = new char[miRNA_sequences.size()][Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH];
	char(*mRNA_cstrings_array_ptr)[Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH] = new char[mRNA_sequences.size()][Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH];
	char(*mRNA_cstrings_reversed_array_ptr)[Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH] = new char[mRNA_sequences.size()][Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH];

	// populate this->miRNA_sequences_chars and this->mRNA_sequences_chars
	StringUtils::convert_strings_to_Cstrings_ptr(miRNA_cstrings_array_ptr, this->miRNA_sequences, miRNA_sequences.size(), Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH);
	StringUtils::convert_strings_to_Cstrings_ptr(mRNA_cstrings_array_ptr, this->mRNA_sequences, mRNA_sequences.size(), Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH);

	this->miRNA_sequences_chars = miRNA_cstrings_array_ptr;
	this->mRNA_sequences_chars = mRNA_cstrings_array_ptr;

	// create and populate int arrays of sequence lengths
	this->miRNA_lengths = new int[miRNA_sequences.size()];
	this->mRNA_lengths = new int[mRNA_sequences.size()];
	count_sequence_lengths(&miRNA_lengths, miRNA_sequences);
	count_sequence_lengths(&mRNA_lengths, mRNA_sequences);
	//StringUtils::print_array(&miRNA_lengths, miRNA_sequences.size(), "Printing computed miRNA lengths:");

	// reverse mRNA sequences
	StringUtils::reverse_array(mRNA_cstrings_reversed_array_ptr, mRNA_cstrings_array_ptr, mRNA_sequences.size(), mRNA_lengths);
	this->mRNA_sequences_chars_reversed = mRNA_cstrings_reversed_array_ptr;

	// this will store the results after the kernel runs
	this->sequence_comparison_results = new float[miRNA_sequences.size() * mRNA_sequences.size()];
	

	std::cout << "Maximum miRNA and mRNA lengths (in nucleotides) are: " << max_miRNA_length << " and " << max_mRNA_length << std::endl;
	printf("\x1B[31mTesting\033[0m\n");
	printf("\x1B[32mTesting\033[0m\n");

	if ((max_miRNA_length >= Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH) || (max_mRNA_length >= Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH)) {
		// if any miRNA or mRNA is longer than Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH, issue a warning.
		cuda_function_safety_condition = false;
		printf("\x1B[91mWARNING:Either one miRNA or mRNA are longer than Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH.You should not use CUDA functions.\033[0m\n");
		//std::cout << "WARNING: Either one miRNA or mRNA are longer than Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH. You should not use CUDA functions." << std::endl;
	}
	else {
		cuda_function_safety_condition = true;
	}
}


CUDASequenceComparator::~CUDASequenceComparator() {
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
std::vector<std::vector<std::string>> CUDASequenceComparator::process_miRNAsequences_file(std::string miRNA_sequences_hsa_only_filepath) {
	std::vector<std::string> miRNA_miRDB_ids;
	std::vector<std::string> miRNA_names;
	std::vector<std::string> miRNA_sequences;

	std::ifstream file(miRNA_sequences_hsa_only_filepath);
	if (file.is_open()) {
		std::string line;
		while (std::getline(file, line)) {
			//std::vector<std::string> line_elements = StringUtils::split(line, "\t");
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
std::vector<std::vector<std::string>> CUDASequenceComparator::process_mRNAsequences_file(std::string mRNA_sequences_filepath) {
	std::vector<std::string> mRNA_uniprot_ids;
	std::vector<std::string> mRNA_sequences;

	std::ifstream file(mRNA_sequences_filepath);
	if (file.is_open()) {
		std::string line;
		int line_index = 0;
		while (std::getline(file, line)) {
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
void CUDASequenceComparator::count_sequence_lengths(int** dst_array, std::vector<std::string> sequences) {
	// todo: check that the sizes of dst_array and sequences match
	for (int i = 0; i < sequences.size(); i++) {
		(*dst_array)[i] = sequences[i].length();
	}
}

/*
 * Executes sequence comparison for this->miRNA_sequences_chars and this->mRNA_sequences_chars
 */
void CUDASequenceComparator::compare_sequences() {
	int numBlocks = miRNA_sequences.size();
	int numThreads = mRNA_sequences.size(); // num threads per each block

	float* d_result_array;
	char(*d_miRNA_sequences)[Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH];
	char(*d_mRNA_sequences)[Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH];
	char(*d_mRNA_sequences_reversed)[Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH];
	int* d_miRNA_lengths;
	int* d_mRNA_lengths;

	// set the primary GPU
	cudaSetDevice(0);

	// allocate GPU memory
	cudaMalloc((void**)&d_result_array, numBlocks * numThreads * sizeof(float)); // there are miRNA_count * mRNA_count results of type int
	cudaMalloc((void**)&d_miRNA_sequences, miRNA_sequences.size() * Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH * sizeof(char)); // 2D char array with miRNA_sequences.size() rows and MAX_CHAR_ARRAY_SEQUENCE_LENGTHS columns
	cudaMalloc((void**)&d_mRNA_sequences, mRNA_sequences.size() * Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH * sizeof(char));
	cudaMalloc((void**)&d_mRNA_sequences_reversed, mRNA_sequences.size() * Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH * sizeof(char));
	cudaMalloc((void**)&d_miRNA_lengths, miRNA_sequences.size() * sizeof(int)); // count of lengths corresponds to the count of miRNAs
	cudaMalloc((void**)&d_mRNA_lengths, mRNA_sequences.size() * sizeof(int));

	// copy data from host to device
	cudaMemcpy(d_miRNA_sequences, this->miRNA_sequences_chars, miRNA_sequences.size() * Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH * sizeof(char), cudaMemcpyHostToDevice);
	cudaMemcpy(d_mRNA_sequences, this->mRNA_sequences_chars, mRNA_sequences.size() * Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH * sizeof(char), cudaMemcpyHostToDevice);
	cudaMemcpy(d_mRNA_sequences_reversed, this->mRNA_sequences_chars_reversed, mRNA_sequences.size() * Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH * sizeof(char), cudaMemcpyHostToDevice);
	cudaMemcpy(d_miRNA_lengths, this->miRNA_lengths, miRNA_sequences.size() * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_mRNA_lengths, this->mRNA_lengths, mRNA_sequences.size() * sizeof(int), cudaMemcpyHostToDevice);

	// run the gpu kernel
	compare_sequence_arrays_kernel<<<numBlocks, numThreads>>>(d_result_array, d_miRNA_sequences, d_mRNA_sequences, d_mRNA_sequences_reversed, d_miRNA_lengths, d_mRNA_lengths);
	cudaDeviceSynchronize();

	// copy back to host
	cudaMemcpy(this->sequence_comparison_results, d_result_array, numBlocks * numThreads * sizeof(float), cudaMemcpyDeviceToHost);

	// test success
	std::cout << "result[0] = " << sequence_comparison_results[0] << std::endl;

	// free device memory
	cudaFree(d_result_array);
	cudaFree(d_miRNA_sequences);
	cudaFree(d_mRNA_sequences);
	cudaFree(d_mRNA_sequences_reversed);
	cudaFree(d_miRNA_lengths);
	cudaFree(d_mRNA_lengths);
}


void CUDASequenceComparator::save_sequence_comparison_results(std::string filepath) {
	std::cout << "Saving sequence comparison results into " << filepath << std::endl;

	std::ofstream outfile(filepath, std::ios_base::app); // open the file in append mode
	if (outfile.is_open()) {
		for (int i = 0; i < miRNA_sequences.size(); i++) {
			// write the current miRNA id without tab indentation
			outfile << miRNA_miRDB_ids[i] << "," << miRNA_names[i] << std::endl;
			for (int j = 0; j < mRNA_sequences.size(); j++) {
				// write all corresponding mRNA matches (and their scores) WITH tab indentation
				// example: unirotkb-xxxx : 0.16
				outfile << "\t" << mRNA_ids[j] << ": " << sequence_comparison_results[i * j] << std::endl;
			}
		}
		outfile.close();
	}
	else {
		std::cout << "Error opening file " << filepath << std::endl;
	}
}