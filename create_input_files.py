from utils import create_input_files

if __name__ == '__main__':
    # Create input files (along with word map)
    create_input_files(dataset='baby',
                       captions_per_image=1,
                       min_word_freq=0,
                       output_folder='/work/06117/lakuduva/maverick2/output')
