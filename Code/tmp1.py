# 修改1：移除batch_size限制（原247行附近）
# 将断言改为条件判断提示
if torch.distributed.get_world_size() > 1 and args.batch_size != 1:
    raise ValueError('Multi-GPU only supports batch_size=1')
elif args.batch_size > 1:
    print(f'启用批量推理: batch_size={args.batch_size}')

# 修改2：重构collate_fn（约42行）
def collate_fn(batches, tokenizer):
    # 将torch.cat改为torch.stack保持批次维度
    pixel_values = torch.stack([_['pixel_values'] for _ in batches])  # [B, K, C, H, W]
    texts = [_['text'] for _ in batches]
    bboxes = [_['bbox'] for _ in batches]
    hws = [_['hw'] for _ in batches]
    return pixel_values, texts, bboxes, hws

# 修改3：重构推理循环（evaluate_chat_model函数内）
for _, (pixel_values, questions, bboxes, hws) in enumerate(tqdm(dataloader)):
    pixel_values = pixel_values.to(torch.bfloat16).cuda()
    answers = []
    
    # 批量处理每个样本
    for i in range(pixel_values.size(0)):  # 遍历批次中的每个样本
        generation_config = {
            "num_beams": args.num_beams,
            "max_new_tokens": 100,
            "min_new_tokens": 1,
            "do_sample": args.temperature > 0,
            "temperature": args.temperature
        }
        # 处理单个样本：像素值[i] + 问题[i]
        pred = model.chat(
            tokenizer=tokenizer,
            pixel_values=pixel_values[i],  # [K, C, H, W]
            question=questions[i],        # 单个字符串
            generation_config=generation_config
        )
        answers.append(pred)
    
    # 保存结果（原结构不变）
    for bbox, hw, answer in zip(bboxes, hws, answers):
        outputs.append({'answer': answer, 'gt_bbox': bbox, 'hw': hw})




# 在__getitem__末尾添加
if self.dynamic_image_size and len(images) < self.max_num:
    # 用空白图像填充到max_num
    pad_images = [images[-1]] * (self.max_num - len(images))
    images.extend(pad_images)

# 单GPU批量运行
torchrun --nproc_per_node=1 eval/refcoco/evaluate_grounding.py \
    --checkpoint $CHECKPOINT \
    --dynamic \
    --batch_size 5  # 新增批次参数