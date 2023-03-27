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

class CUDASequenceComparatorV1 
{
	std::string miRNAs_filepath;
	std::string mRNAs_filepath;

	std::vector<std::string> miRNA_sequences; // contains all of the miRNA sequences
	std::vector<std::string> miRNA_ids; // contains all of the respective miRNA miRDB tags (eg MI000001)
	std::vector<std::string> miRNA_names; // contains all of the respective miRNA ids (eg. hsa-let-7a-1)
	char* miRNA_sequences_chars;
	int* miRNA_lengths;

	std::vector<std::string> mRNA_sequences; // contains all of the mRNA sequences
	std::vector<std::string> mRNA_ids; // contains uniprot mRNA ids
	std::vector<std::vector<std::string>> mRNA_refseq_ids; //contains all of the NCBI Nucleotide refseq ids
	char* mRNA_sequences_chars;
	char* mRNA_sequences_chars_reversed;
	int* mRNA_lengths; // an array of lengths of each mRNA sequence in mRNA sequences

	int max_miRNA_length; // this is miRNA pitch for CUDA !!!
	int max_mRNA_length; // this is mRNA pitch for CUDA !!!

	float* sequence_comparison_results;

public:
	CUDASequenceComparatorV1(std::string, std::string);
	~CUDASequenceComparatorV1();
	void compare_sequences();
private:
	std::vector<std::vector<std::string>> process_miRNAsequences_file(std::string); // processes mirbase_miRNA_hsa-only.txt
	std::vector<std::vector<std::string>> process_mRNAsequences_file(std::string);
	void count_sequence_lengths(int*&, std::vector<std::string>);
	//int get_element_index(std::vector<std::string>, std::string);
};