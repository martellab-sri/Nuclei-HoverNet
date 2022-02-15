python run_infer.py \
--gpu='5,6,7,3,4' \
--nr_types=5 \
--type_info_path=type_info1.json \
--batch_size=32 \
--model_mode=original \
--model_path=/labs3/amartel_data3/tingxiao/logs_AML_new/01/net_epoch=50.tar \
--nr_inference_workers=18 \
--nr_post_proc_workers=24 \
wsi \
--input_dir=/labs3/amartel_data3/tingxiao/hover_net/dataset/sample_wsis/wsi_mds \
--output_dir=/labs3/amartel_data3/tingxiao/hover_net/dataset/sample_wsis/out_mds/ \
--input_mask_dir=/labs3/amartel_data3/tingxiao/hover_net/dataset/sample_wsis/wsi_mds/mask/ \
--cache_path=/labs3/amartel_data3/tingxiao/hover_net/dataset/sample_wsis/cache/ \
--save_thumb \
--save_mask
