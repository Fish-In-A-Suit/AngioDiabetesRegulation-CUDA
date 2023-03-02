#include "CUDASequenceComparator.cuh"

__global__ void compareSequences(int* d_result_array, char(*d_miRNA_sequences)[Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH], char(*d_mRNA_sequences)[Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH]) {
	printf("Running parallel thread [%d,%d]\n", threadIdx.x, blockIdx.x);
	// char current_miRNA[][Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH] = *(*(d_miRNA_sequences + 0) + 0); // not allowed
	char firstChar_miRNA = *(*(d_miRNA_sequences + 0) + 0);
	char* firstRow_miRNA = d_miRNA_sequences[0];
	char firstCharv1_miRNA = firstRow_miRNA[0];

	char firstChar_mRNA = *(*(d_mRNA_sequences + 0) + 0);
	char* firstRow_mRNA = d_mRNA_sequences[0];
	char firstCharv1_mRNA = firstRow_mRNA[0];

	if (firstChar_mRNA == firstChar_miRNA) {
		d_result_array[0] = 1;
	}
	else {
		d_result_array[0] = 0;
	}
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

	// populate this->miRNA_sequences_chars and this->mRNA_sequences_chars
	StringUtils::convert_strings_to_Cstrings_ptr(miRNA_cstrings_array_ptr, this->miRNA_sequences, miRNA_sequences.size(), Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH);
	StringUtils::convert_strings_to_Cstrings_ptr(mRNA_cstrings_array_ptr, this->mRNA_sequences, mRNA_sequences.size(), Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH);

	this->miRNA_sequences_chars = miRNA_cstrings_array_ptr;
	this->mRNA_sequences_chars = mRNA_cstrings_array_ptr;

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