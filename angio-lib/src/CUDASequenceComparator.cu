#include "CUDASequenceComparator.cuh"

/*
 * Compares two sequences (anneals a miRNA onto a mRNA from anneal_start_index forwards) and returns the match_strength (no_matches / size of miRNA). Can be reused across
 * multiple kernels.
 * 
 * __forceinline__ due to debugging reasons (https://docs.nvidia.com/gameworks/content/developertools/desktop/troubleshooting.htm)
 */
__device__ __forceinline__ float compare_sequences_kernel(char* miRNA_sequence, char* mRNA_sequence, int miRNA_len, int mRNA_len, int anneal_start_index) {
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
		char current_miRNA_char = miRNA_sequence[i-anneal_start_index]; // BUGFIX: shouldn't be 'i' here, since miRNA only goes from 0 to len(miRNA_sequence), whereas the current i iterations are based on mRNA sequence and can exceed the miRNA length; resolution: subtract anneal_start_index
		char current_mRNA_char = mRNA_sequence[i];
		
		if (
			((current_miRNA_char == 'A') && (current_mRNA_char == 'T')) || 
			((current_miRNA_char == 'T') && (current_mRNA_char == 'A')) ||
			((current_miRNA_char == 'C') && (current_mRNA_char == 'G')) ||
			((current_miRNA_char == 'G') && (current_mRNA_char == 'C'))
			) {
			successful_matches++;
		}
	}
	// printf("  - successful_matches = %f, all_characters = %f", (float)successful_matches, (float)all_characters); // error when casting here !!!

	return (float)successful_matches / all_characters;
}

/*
 * A device (GPU) implementation of the std::strlen function to find the number of characters inside a const char* cstr.
 */
__device__ __forceinline__ int my_strlen_dev(char* cstr) {
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
__device__ __forceinline__ void debug_print_sequence_v1(char* sequence, int char_count) {
	int start_char_index = 0;
	int end_char_index = my_strlen_dev(sequence);

	printf("%.*s", char_count, sequence + start_char_index); // print first char_count elements
	printf(" ... ");
	printf(".*s\n", char_count, sequence + end_char_index - char_count); // print last char_count elements
}

__device__ __forceinline__ void debug_print_sequence_v2(char* sequence, int char_count, int seq_length) {
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
__device__ __forceinline__ char* reverse_char_array(char* char_array_to_flip, int array_len) {
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
	// --- *** end of test *** ---

	// --- *** MAIN LOGIC *** ---
	char* current_miRNA = d_miRNA_sequences[blockIdx.x];
	char* current_mRNA = d_mRNA_sequences[threadIdx.x];
	char* current_mRNA_reversed = d_mRNA_sequences_reversed[threadIdx.x]; // BUGFIX_ d_mRNA_sequences_reversed* instead of d_mRNA_sequences
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
	printf("Parallel thread [%d,%d] result at index %d: %f\n", blockIdx.x, threadIdx.x, result_index, max_match_strength);
	d_result_array[result_index] = max_match_strength;
}

__global__ void compare_sequence_arrays_kernel_debug(float* d_result_array, char(*d_miRNA_sequences)[Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH], char(*d_mRNA_sequences)[Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH], char(*d_mRNA_sequences_reversed)[Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH], int* d_miRNA_lengths, int* d_mRNA_lengths, char *d_miRNA_ids, int miRNAids_pitch, char(*d_mRNA_ids)[Constants::MRNA_UNIPROT_ID_LENGTH]) {
	char* current_miRNA = d_miRNA_sequences[blockIdx.x];
	char* current_mRNA = d_mRNA_sequences[threadIdx.x];
	char* current_mRNA_reversed = d_mRNA_sequences_reversed[threadIdx.x]; // BUGFIX_ d_mRNA_sequences_reversed* instead of d_mRNA_sequences
	int current_miRNA_length = d_miRNA_lengths[blockIdx.x];
	int current_mRNA_length = d_mRNA_lengths[threadIdx.x];
	int result_index = blockIdx.x * blockDim.x + threadIdx.x; // blockDim.x represents the number of threads in a block (= the number of mRNAs)

	char* current_miRNA_row_ptr = (char*)((char*)d_miRNA_ids + blockIdx.x * miRNAids_pitch); // 10 is the size of a miRNA MI id
	printf("%.*s\n", miRNAids_pitch, current_miRNA_row_ptr);
	
	
	char* current_mRNA_id = d_mRNA_ids[threadIdx.x];

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
	// printf("[blockIdx, blockDim, threadIdx] = [%d,%d,%d] result at i %d (%s, :: $s): %f\n", blockIdx.x, blockDim.x, threadIdx.x, result_index, current_miRNA_id, current_mRNA_id, max_match_strength);
	d_result_array[result_index] = max_match_strength;
}

/*
  se = "start-end"
  Compares miRNA and mRNA sequence by inputting entire arrays and specifying start indices and lengths

  Should test this with compare_sequence_arrays_kernel_v2
*/
__device__ __forceinline__ float compare_sequences_kernel_se(char* miRNA_sequences_array, char* mRNA_sequences_array, int miRNA_start_index, int miRNA_len, int mRNA_start_index, int mRNA_len, int anneal_start_index) {
	int successful_matches = 0;
	int all_characters = miRNA_len;

	// miRNA_len should be less than mRNA_len
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

		if (
			((current_miRNA_char == 'A') && (current_mRNA_char == 'T')) ||
			((current_miRNA_char == 'T') && (current_mRNA_char == 'A')) ||
			((current_miRNA_char == 'C') && (current_mRNA_char == 'G')) ||
			((current_miRNA_char == 'G') && (current_mRNA_char == 'C'))
			) {
			successful_matches++;
		}
	}
	return (float)successful_matches / all_characters;
}

// for debugging purposes, this function only returns operation count
__device__ __forceinline__ int compare_sequences_kernel_se_opcounts(char* miRNA_sequences_array, char* mRNA_sequences_array, int miRNA_start_index, int miRNA_len, int mRNA_start_index, int mRNA_len, int anneal_start_index) {
	int successful_matches = 0;
	int all_characters = miRNA_len;

	// miRNA_len should be less than mRNA_len
	if (miRNA_len > mRNA_len) {
		printf("miRNA_len > mRNA_len, returning -1");
		return -1;
	}

	int opcount = 0;
	int mRNA_nucleotide_anneal_start = mRNA_start_index + anneal_start_index;
	for (int i = mRNA_nucleotide_anneal_start; i < (mRNA_nucleotide_anneal_start + miRNA_len); i++) {
		if (i > mRNA_start_index + mRNA_len) {
			break;
		}
		char current_miRNA_char = miRNA_sequences_array[i - mRNA_nucleotide_anneal_start + miRNA_start_index]; // start at miRNA_start_index, then go by 1 all the way until miRNA_len
		char current_mRNA_char = mRNA_sequences_array[i];

		if (
			((current_miRNA_char == 'A') && (current_mRNA_char == 'T')) ||
			((current_miRNA_char == 'T') && (current_mRNA_char == 'A')) ||
			((current_miRNA_char == 'C') && (current_mRNA_char == 'G')) ||
			((current_miRNA_char == 'G') && (current_mRNA_char == 'C'))
			) {
			successful_matches++;
		}
		opcount++;
	}
	return opcount;
}

/*
  ptr = "pointer"
  Compares miRNA and mRNA sequences by inputting row pointers (single miRNA and mRNA sequence) and lengths

  Should test this with compare_sequence_arrays_kernel_v2
*/
__device__ __forceinline__ float compare_sequences_kernel_ptr(char* d_miRNA_sequence, char* d_mRNA_sequence, int miRNA_len, int mRNA_len, int anneal_start_index) {
	// TODO
}

__global__ void compare_sequence_arrays_kernel_v2(float* d_result_array, char* d_miRNA_sequences_array, char* d_mRNA_sequences_array, char* d_mRNA_sequences_reversed_array, int miRNA_seqarr_pitch, int mRNA_seqarr_pitch, int* d_miRNA_lengths, int* d_mRNA_lengths) {
	// blockIdx.x determines miRNA
	// threadIdx.x determines mRNA
	// blockDim.x = count mRNAs (amount of threads in a single block)
	int miRNA_start_index = blockIdx.x * miRNA_seqarr_pitch;
	char* current_miRNA_row_ptr = (char*)((char*)d_miRNA_sequences_array + miRNA_start_index); // since d_miRNA_sequences_array is a 1D array, this goes up to miRNA_start_index into the array
	int mRNA_start_index = threadIdx.x * mRNA_seqarr_pitch;
	char* current_mRNA_row_ptr = (char*)((char*)d_mRNA_sequences_array + mRNA_start_index);
	char* current_mRNA_rev_row_ptr = (char*)((char*)d_mRNA_sequences_reversed_array + mRNA_start_index);

	int miRNA_length = d_miRNA_lengths[blockIdx.x];
	int mRNA_length = d_mRNA_lengths[threadIdx.x];

	int result_index = blockIdx.x * blockDim.x + threadIdx.x;
	float max_match_strength = 0.0f;

	for (int i = 0; i < mRNA_length - miRNA_length; i++) 
	{
		float match_strength_straight = compare_sequences_kernel_se(d_miRNA_sequences_array, d_mRNA_sequences_array, miRNA_start_index, miRNA_length, mRNA_start_index, mRNA_length, i);
		float match_strength_reversed = compare_sequences_kernel_se(d_miRNA_sequences_array, d_mRNA_sequences_reversed_array, miRNA_start_index, miRNA_length, mRNA_start_index, mRNA_length, i);
		
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

__global__ void compare_sequence_arrays_kernel_v2_opcounts(float* d_result_array, char* d_miRNA_sequences_array, char* d_mRNA_sequences_array, char* d_mRNA_sequences_reversed_array, int miRNA_seqarr_pitch, int mRNA_seqarr_pitch, int* d_miRNA_lengths, int* d_mRNA_lengths) {
	int miRNA_start_index = blockIdx.x * miRNA_seqarr_pitch;
	char* current_miRNA_row_ptr = (char*)((char*)d_miRNA_sequences_array + miRNA_start_index); // since d_miRNA_sequences_array is a 1D array, this goes up to miRNA_start_index into the array
	int mRNA_start_index = threadIdx.x * mRNA_seqarr_pitch;
	char* current_mRNA_row_ptr = (char*)((char*)d_mRNA_sequences_array + mRNA_start_index);
	char* current_mRNA_rev_row_ptr = (char*)((char*)d_mRNA_sequences_reversed_array + mRNA_start_index);

	int miRNA_length = d_miRNA_lengths[blockIdx.x];
	int mRNA_length = d_mRNA_lengths[threadIdx.x];

	int result_index = blockIdx.x * blockDim.x + threadIdx.x;
	float max_match_strength = 0.0f;

	int opcount = 0;
	int max = 0;
	for (int i = 0; i < mRNA_length - miRNA_length; i++)
	{
		opcount += compare_sequences_kernel_se_opcounts(d_miRNA_sequences_array, d_mRNA_sequences_array, miRNA_start_index, miRNA_length, mRNA_start_index, mRNA_length, i);
		// float match_strength_reversed = compare_sequences_kernel_se(d_miRNA_sequences_array, d_mRNA_sequences_reversed_array, miRNA_start_index, miRNA_length, mRNA_start_index, mRNA_length, i);
		/*
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
		*/
		if (i > max) {
			max = i;
		}
	}

	printf("Parallel thread [%d,%d]: miRNA_len = %d, mRNA_len = %d, (diff = %d): opcount_total = %d, opcount_sub = %d\n", blockIdx.x, threadIdx.x, miRNA_length, mRNA_length, mRNA_length-miRNA_length, opcount, (int) opcount/max);
	d_result_array[result_index] = max_match_strength;
}

__global__ void copy_kernel_2D(float* devPtr, size_t pitch, int width, int height) {
	for (int r = 0; r < height; ++r) {
		float* row = (float*)((char*)devPtr + r * pitch);
		for (int c = 0; c < width; ++c) {
			float element = row[c];
		}
	}
}

__global__ void miRNA_id_copy_kernel(char* miRNAids, int pitch) {
	char* row_ptr = (char*)((char*)miRNAids + blockIdx.x * pitch);
	// printf("%.*s\n", pitch, row_ptr); // this notation doesnt work
	printf("Parallel thread [%d,%d]:    ", blockIdx.x, threadIdx.x);
	for (int i = 0; i < pitch; i++) {
		printf("%c", row_ptr[i]);
	}
	printf("\n");


	//t
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
	std::cout << "Processing miRNA_sequences_chars and mRNA_sequences_chars." << std::endl;
	StringUtils::convert_strings_to_Cstrings_ptr(miRNA_cstrings_array_ptr, this->miRNA_sequences, miRNA_sequences.size(), Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH);
	StringUtils::convert_strings_to_Cstrings_ptr(mRNA_cstrings_array_ptr, this->mRNA_sequences, mRNA_sequences.size(), Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH);
	this->miRNA_sequences_chars = miRNA_cstrings_array_ptr;
	this->mRNA_sequences_chars = mRNA_cstrings_array_ptr;

	// populate this->_debug_miRNA_ids and this->_debug_mRNA_ids
	std::cout << "Processing _debug_miRNA_ids and _debug_mRNA_ids" << std::endl;
	char(*miRNAids_cstrings_array_ptr)[Constants::MIRNA_MI_ID_LENGTH] = new char[miRNA_miRDB_ids.size()][Constants::MIRNA_MI_ID_LENGTH];
	char(*mRNAids_cstrings_array_ptr)[Constants::MRNA_UNIPROT_ID_LENGTH] = new char[mRNA_ids.size()][Constants::MRNA_UNIPROT_ID_LENGTH]; 
	StringUtils::convert_strings_to_Cstrings_ptr(miRNAids_cstrings_array_ptr, this->miRNA_miRDB_ids, miRNA_miRDB_ids.size(), Constants::MIRNA_MI_ID_LENGTH);
	StringUtils::convert_strings_to_Cstrings_ptr(mRNAids_cstrings_array_ptr, this->mRNA_ids, mRNA_ids.size(), Constants::MRNA_UNIPROT_ID_LENGTH);
	this->_debug_miRNA_ids = miRNAids_cstrings_array_ptr;
	this->_debug_mRNA_ids = mRNAids_cstrings_array_ptr;

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
	cudaError_t cudaerr = cudaDeviceSynchronize();
	if (cudaerr != cudaSuccess)
		printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));

	// copy back to host
	cudaMemcpy(this->sequence_comparison_results, d_result_array, numBlocks * numThreads * sizeof(float), cudaMemcpyDeviceToHost);

	// test success
	// std::cout << "result[0] = " << sequence_comparison_results[0] << std::endl;

	// free device memory
	cudaFree(d_result_array);
	cudaFree(d_miRNA_sequences);
	cudaFree(d_mRNA_sequences);
	cudaFree(d_mRNA_sequences_reversed);
	cudaFree(d_miRNA_lengths);
	cudaFree(d_mRNA_lengths);
}

void CUDASequenceComparator::compare_sequences_v2() {
	int numBlocks = miRNA_sequences.size();
	int numThreads = mRNA_sequences.size(); // num threads per each block

	char* d_miRNA_sequences;
	size_t pitch_miRNA_sequences;
	char* d_mRNA_sequences;
	size_t pitch_mRNA_sequences;
	char* d_mRNA_sequences_reversed;
	
	float* d_result_array;
	int* d_miRNA_lengths;
	int* d_mRNA_lengths;

	// miRNA setup
	char* h_miRNA_sequences = (char*)malloc(sizeof(char) * this->miRNA_sequences.size() * Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH);
	for (int i = 0; i < this->miRNA_sequences.size(); i++) { // linearise miRNA_sequences into h_miRNA_sequences
		for (int j = 0; j < Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH; j++) {
			// h_miRNA_sequences[i * Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH + j] = this->miRNA_sequences[i][j];
			h_miRNA_sequences[i * Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH + j] = this->miRNA_sequences_chars[i][j];
		}
	}
	
	// mRNA setup
	char* h_mRNA_sequences = (char*)malloc(sizeof(char) * this->mRNA_sequences.size() * Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH);
	for (int i = 0; i < this->mRNA_sequences.size(); i++) {
		for (int j = 0; j < Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH; j++) {
			// h_mRNA_sequences[i * Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH + j] = this->mRNA_sequences[i][j];
			h_mRNA_sequences[i * Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH + j] = this->mRNA_sequences_chars[i][j];
		}
	}
	
	// mRNA reverse setup
	char* h_mRNA_sequences_reversed = (char*)malloc(sizeof(char) * this->mRNA_sequences.size() * Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH);
	for (int i = 0; i < this->mRNA_sequences.size(); i++) {
		for (int j = 0; j < Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH; j++) {
			h_mRNA_sequences_reversed[i * Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH + j] = this->mRNA_sequences_chars_reversed[i][j];
		}
	}

	// set the primary GPU
	cudaSetDevice(0);

	// device allocations
	cudaMalloc((void**)&d_result_array, numBlocks * numThreads * sizeof(float)); // there are miRNA_count * mRNA_count results of type int
	/* TODO: check if 2D allocation is even necessary if we are using linearised arrays
	cudaMallocPitch(&d_miRNA_sequences, &pitch_miRNA_sequences, Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH * sizeof(char), this->miRNA_sequences.size());
	cudaMallocPitch(&d_mRNA_sequences, &pitch_mRNA_sequences, Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH * sizeof(char), this->mRNA_sequences.size());

	cudaMemcpy2D(d_miRNA_sequences, pitch_miRNA_sequences, h_miRNA_sequences, Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH * sizeof(char), Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH * sizeof(char), this->miRNA_sequences.size(), cudaMemcpyHostToDevice);
	cudaMemcpy2D(d_mRNA_sequences, pitch_mRNA_sequences, h_mRNA_sequences, Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH * sizeof(char), Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH * sizeof(char), this->mRNA_sequences.size(), cudaMemcpyHostToDevice);
	*/
	cudaMalloc((void**)&d_miRNA_sequences, miRNA_sequences.size() * Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH * sizeof(char)); // 2D char array with miRNA_sequences.size() rows and MAX_CHAR_ARRAY_SEQUENCE_LENGTHS columns
	cudaMalloc((void**)&d_mRNA_sequences, mRNA_sequences.size() * Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH * sizeof(char));
	cudaMalloc((void**)&d_mRNA_sequences_reversed, mRNA_sequences.size() * Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH * sizeof(char));
	cudaMalloc((void**)&d_miRNA_lengths, miRNA_sequences.size() * sizeof(int)); // count of lengths corresponds to the count of miRNAs
	cudaMalloc((void**)&d_mRNA_lengths, mRNA_sequences.size() * sizeof(int));

	// copy data to device
	cudaMemcpy(d_miRNA_sequences, h_miRNA_sequences, miRNA_sequences.size() * Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH * sizeof(char), cudaMemcpyHostToDevice);
	cudaMemcpy(d_mRNA_sequences, h_mRNA_sequences, mRNA_sequences.size() * Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH * sizeof(char), cudaMemcpyHostToDevice);
	cudaMemcpy(d_mRNA_sequences_reversed, h_mRNA_sequences_reversed, mRNA_sequences.size() * Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH * sizeof(char), cudaMemcpyHostToDevice);
	cudaMemcpy(d_miRNA_lengths, this->miRNA_lengths, miRNA_sequences.size() * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_mRNA_lengths, this->mRNA_lengths, mRNA_sequences.size() * sizeof(int), cudaMemcpyHostToDevice);

	// run the gpu kernel
	// NOTE: if you used cudaMallocPitch, then you should supply appropriate pitches to the kernel!!!!
	// compare_sequence_arrays_kernel_v2(float* d_result_array, char* d_miRNA_sequences_array, char* d_mRNA_sequences_array, char* d_mRNA_sequences_reversed_array, int miRNA_seqarr_pitch, int mRNA_seqarr_pitch, int* d_miRNA_lengths, int* d_mRNA_lengths)
	
	// this works:
	// compare_sequence_arrays_kernel_v2<<<numBlocks, numThreads>>>(d_result_array, d_miRNA_sequences, d_mRNA_sequences, d_mRNA_sequences_reversed, Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH, Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH, d_miRNA_lengths, d_mRNA_lengths);
	// debug only, delete bottom line
	compare_sequence_arrays_kernel_v2_opcounts<<<numBlocks, numThreads >>>(d_result_array, d_miRNA_sequences, d_mRNA_sequences, d_mRNA_sequences_reversed, Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH, Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH, d_miRNA_lengths, d_mRNA_lengths);

	cudaDeviceSynchronize();

	// copy back
	cudaMemcpy(this->sequence_comparison_results, d_result_array, numBlocks * numThreads * sizeof(float), cudaMemcpyDeviceToHost);

	// free:
	cudaFree(d_result_array);
	cudaFree(d_miRNA_sequences);
	cudaFree(d_mRNA_sequences);
	cudaFree(d_mRNA_sequences_reversed);
	cudaFree(d_miRNA_lengths);
	cudaFree(d_mRNA_lengths);

}

/**
 * This function compares sequences, but with additional miRNA and mRNA ids copied to the GPU for easier debug analysis.
*/
void CUDASequenceComparator::compare_sequences_debug() {
	// set the primary GPU
	cudaSetDevice(0);

	int numBlocks = miRNA_sequences.size();
	int numThreads = mRNA_sequences.size(); // num threads per each block

	float* d_result_array;
	char(*d_miRNA_sequences)[Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH];
	char(*d_mRNA_sequences)[Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH];
	char(*d_mRNA_sequences_reversed)[Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH];
	int* d_miRNA_lengths;
	int* d_mRNA_lengths;

	/* ATTEMPT at MI ids v1
	char* h_debug_miRNA_ids;
	char* d_debug_miRNA_ids;
	h_debug_miRNA_ids = (char*)malloc(sizeof(char) * Constants::MIRNA_MI_ID_LENGTH * this->miRNA_miRDB_ids.size());
	for (int i = 0; i < miRNA_miRDB_ids.size(); i++) { // convert this->miRNA_miRDB_ids into a linear array
		for (int j = 0; j < Constants::MIRNA_MI_ID_LENGTH-1; j++) {
			h_debug_miRNA_ids[i * Constants::MIRNA_MI_ID_LENGTH + j] = miRNA_miRDB_ids[i][j];
		}
	}
	size_t pitch_miRNAids;
	cudaMallocPitch(&d_debug_miRNA_ids, &pitch_miRNAids, sizeof(char) * Constants::MIRNA_MI_ID_LENGTH, miRNA_miRDB_ids.size());
	cudaMemcpy2D(d_debug_miRNA_ids, pitch_miRNAids, h_debug_miRNA_ids, sizeof(char) * Constants::MIRNA_MI_ID_LENGTH, sizeof(char) * Constants::MIRNA_MI_ID_LENGTH, miRNA_miRDB_ids.size(), cudaMemcpyHostToDevice);
	*/
	char(*d_debug_mRNA_ids)[Constants::MRNA_UNIPROT_ID_LENGTH];

	int w = 10;
	int h = 5;
	char* h_debug_miRNA_ids = (char*)malloc(sizeof(char) * w * h);
	for (int i = 0; i < w * h; i++) {
		h_debug_miRNA_ids[i] = 'A' + i % 26; // populate with some random characters
	}
	char* d_debug_miRNA_ids;
	size_t pitch_miRNAids;
	cudaMallocPitch(&d_debug_miRNA_ids, &pitch_miRNAids, sizeof(char) * w, h);
	cudaMemcpy2D(d_debug_miRNA_ids, pitch_miRNAids, h_debug_miRNA_ids, w * sizeof(char), w * sizeof(char), h, cudaMemcpyHostToDevice);

	/*
	// TEST ATTEMPT v1: copying 2D char array to device
	// declare host and device data
	char miRNAidchars[3][10] = { {'M','I','0','0','0','0','0','0','1','\0'},{'M','I','0','0','0','0','0','0','2','\0'},{'M','I','0','0','0','0','0','0','3','\0'} };
	char device_data[3][10];
	// allocate memory on the device
	char* dev_ptr;
	cudaMalloc((void**)&dev_ptr, sizeof(char) * 3 * 10);
	// set the dimensions and pitch of the source and destination arrays
	int src_pitch = 10 * sizeof(char);
	int dst_pitch = 10 * sizeof(char);
	int width = 10;
	int height = 3;
	// copy data from host to device
	cudaMemcpy2D(dev_ptr, dst_pitch, miRNAidchars, src_pitch, width * sizeof(char), height, cudaMemcpyHostToDevice);
	// copy data back
	cudaMemcpy2D(device_data, src_pitch, dev_ptr, dst_pitch, width * sizeof(char), height, cudaMemcpyDeviceToHost);
	// print out the data
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			printf("%s", device_data[i][j]);
		}
		printf("\n");
	}
	// free
	cudaFree(dev_ptr);
	// END TEST v1
	*/

	// TEST ATTEMPT V2: works
	float* A, * d_A;
	size_t pitch;
	int width = 10;
	int height = 3;
	float* results;

	A = (float*)malloc(sizeof(float) * width * height); // this is a 1D linear array!!!
	results = (float*)malloc(sizeof(float) * width * height);
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			A[i * width + j] = i + j;
		}
	}
	cudaMallocPitch(&d_A, &pitch, sizeof(float) * width, height);
	cudaMemcpy2D(d_A, pitch, A, sizeof(float) * width, sizeof(float) * width, height, cudaMemcpyHostToDevice);
	cudaMemcpy2D(results, sizeof(float)*width, d_A, pitch, sizeof(float) * width, height, cudaMemcpyDeviceToHost);
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			printf("%f,", results[i*width+j]);
		}
		printf("\n");
	}
	// END TEST v2

	// allocate GPU memory
	cudaMalloc((void**)&d_result_array, numBlocks * numThreads * sizeof(float)); // there are miRNA_count * mRNA_count results of type int
	cudaMalloc((void**)&d_miRNA_sequences, miRNA_sequences.size() * Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH * sizeof(char)); // 2D char array with miRNA_sequences.size() rows and MAX_CHAR_ARRAY_SEQUENCE_LENGTHS columns
	cudaMalloc((void**)&d_mRNA_sequences, mRNA_sequences.size() * Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH * sizeof(char));
	cudaMalloc((void**)&d_mRNA_sequences_reversed, mRNA_sequences.size() * Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH * sizeof(char));
	cudaMalloc((void**)&d_miRNA_lengths, miRNA_sequences.size() * sizeof(int)); // count of lengths corresponds to the count of miRNAs
	cudaMalloc((void**)&d_mRNA_lengths, mRNA_sequences.size() * sizeof(int));

	// cudaMalloc((void**)&d_debug_miRNA_ids, miRNA_miRDB_ids.size() * Constants::MIRNA_MI_ID_LENGTH * sizeof(char));
	cudaMalloc((void**)&d_debug_mRNA_ids, mRNA_ids.size() * Constants::MRNA_UNIPROT_ID_LENGTH * sizeof(char)); 

	// copy data from host to device
	cudaMemcpy(d_miRNA_sequences, this->miRNA_sequences_chars, miRNA_sequences.size() * Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH * sizeof(char), cudaMemcpyHostToDevice);
	cudaMemcpy(d_mRNA_sequences, this->mRNA_sequences_chars, mRNA_sequences.size() * Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH * sizeof(char), cudaMemcpyHostToDevice);
	cudaMemcpy(d_mRNA_sequences_reversed, this->mRNA_sequences_chars_reversed, mRNA_sequences.size() * Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH * sizeof(char), cudaMemcpyHostToDevice);
	cudaMemcpy(d_miRNA_lengths, this->miRNA_lengths, miRNA_sequences.size() * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_mRNA_lengths, this->mRNA_lengths, mRNA_sequences.size() * sizeof(int), cudaMemcpyHostToDevice);

	// cudaMemcpy(d_debug_miRNA_ids, this->_debug_miRNA_ids, miRNA_sequences.size() * Constants::MIRNA_MI_ID_LENGTH * sizeof(char), cudaMemcpyHostToDevice);
	// cudaMemcpy2D(d_debug_miRNA_ids, Constants::MIRNA_MI_ID_LENGTH * sizeof(char), this->_debug_miRNA_ids, Constants::MIRNA_MI_ID_LENGTH * sizeof(char), Constants::MIRNA_MI_ID_LENGTH * sizeof(char), miRNA_sequences.size(), cudaMemcpyHostToDevice);
	
	cudaMemcpy(d_debug_mRNA_ids, this->_debug_mRNA_ids, mRNA_sequences.size() * Constants::MRNA_UNIPROT_ID_LENGTH * sizeof(char), cudaMemcpyHostToDevice);

	// run the gpu kernel TODO: enable the top one !!!
	// compare_sequence_arrays_kernel_debug<<<numBlocks, numThreads >>>(d_result_array, d_miRNA_sequences, d_mRNA_sequences, d_mRNA_sequences_reversed, d_miRNA_lengths, d_mRNA_lengths, d_debug_miRNA_ids, d_debug_mRNA_ids);
	// compare_sequence_arrays_kernel_debug <<<numBlocks, 1>>> (d_result_array, d_miRNA_sequences, d_mRNA_sequences, d_mRNA_sequences_reversed, d_miRNA_lengths, d_mRNA_lengths, d_debug_miRNA_ids, pitch_miRNAids, d_debug_mRNA_ids);
	miRNA_id_copy_kernel <<<h, 1 >>>(d_debug_miRNA_ids, pitch_miRNAids);
	cudaDeviceSynchronize();

	// copy back to host
	cudaMemcpy(this->sequence_comparison_results, d_result_array, numBlocks * numThreads * sizeof(float), cudaMemcpyDeviceToHost);


	// free device memory
	cudaFree(d_result_array);
	cudaFree(d_miRNA_sequences);
	cudaFree(d_mRNA_sequences);
	cudaFree(d_mRNA_sequences_reversed);
	cudaFree(d_miRNA_lengths);
	cudaFree(d_mRNA_lengths);
	cudaFree(d_debug_miRNA_ids);
	cudaFree(d_debug_mRNA_ids);
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
				// example: unirotkb-xxxx : 0.175
				outfile << "\t" << mRNA_ids[j] << ": " << sequence_comparison_results[i * j] << std::endl;
			}
		}
		outfile.close();
	}
	else {
		std::cout << "Error opening file " << filepath << std::endl;
	}
}