#pragma once

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

#include "StringUtils.h"
#include "Constants.h"

class CUDASequenceComparator
{
	std::string miRNAs_filepath;
	std::string mRNAs_filepath;

	std::vector<std::string> miRNA_sequences; // contains all of the miRNA sequences
	std::vector<std::string> miRNA_miRDB_ids; // contains all of the respective miRNA miRDB tags (eg MI000001)
	std::vector<std::string> miRNA_names; // contains all of the respective miRNA ids (eg. hsa-let-7a-1)
	char (*miRNA_sequences_chars)[Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH];
	int* miRNA_lengths; // an array of lengths of each miRNA sequence in miRNA_sequences
	
	std::vector<std::string> mRNA_sequences; // contains all of the mRNA sequences
	std::vector<std::string> mRNA_ids; // contains uniprot mRNA ids
	std::vector<std::vector<std::string>> mRNA_refseq_ids; //contains all of the NCBI Nucleotide refseq ids
	char(*mRNA_sequences_chars)[Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH];
	int* mRNA_lengths; // an array of lengths of each mRNA sequence in mRNA sequences

	int max_miRNA_length;
	int max_mRNA_length;

	float* sequence_comparison_results;

	bool cuda_function_safety_condition = false; // set this to true when it is checked that no miRNA or mRNA is longer than the value set in Constants

public:
	CUDASequenceComparator(std::string, std::string);
	~CUDASequenceComparator();
	void compare_sequences();

private:
	std::vector<std::vector<std::string>> process_miRNAsequences_file(std::string); // processes mirbase_miRNA_hsa-only.txt
	std::vector<std::vector<std::string>> process_mRNAsequences_file(std::string);
	void count_sequence_lengths(int**, std::vector<std::string>);
};
