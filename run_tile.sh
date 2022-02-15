python run_infer.py \
--gpu='4,5,6,7' \
--nr_types=5 \
--type_info_path=type_info1.json \
--batch_size=32 \
--model_mode=original \
--model_path=/labs3/amartel_data3/tingxiao/logs_AML_new/01/net_epoch=50.tar \
--nr_inference_workers=16 \
--nr_post_proc_workers=16 \
tile \
--input_dir=/labs3/amartel_data3/tingxiao/hover_net/dataset/sample_tiles/MDS_model_ROI/ \
--output_dir=/labs3/amartel_data3/tingxiao/hover_net/dataset/sample_tiles/pred_mds_model_ROI/ \
--mem_usage=0.1 \
--save_qupath
