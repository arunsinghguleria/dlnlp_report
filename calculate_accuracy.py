


def calculate_accuracy(model):

    dataset = load_dataset("cnn_dailymail", "3.0.0")
    train_data = dataset["train"]
    val_data = dataset["validation"]

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    t5_rouge1Score = [0,0,0]
    t5_rouge2Score = [0,0,0]
    t5_rougeLScore = [0,0,0]




    # model_path = "./t5_finetuned_cnn_dailymail" # model path
    # tokenizer = T5Tokenizer.from_pretrained(model_path)
    # model = T5ForConditionalGeneration.from_pretrained(model_path)


    for i in range(val_data.num_rows):
        text,label = val_data[i]['article'],val_data[i]['highlights']



        # Prefix the input text for summarization
        inputs = [text]

        tokenized_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="longest", return_tensors="pt")

        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            summary_ids = model.generate(
                input_ids=tokenized_inputs["input_ids"],
                attention_mask=tokenized_inputs["attention_mask"],
                max_length=150,
                num_beams=4,  # Use beam search for better summaries
                length_penalty=2.0,  # Penalize overly long summaries
                early_stopping=True
            )

        # Decode the generated summaries
        summary = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]

        scores = scorer.score(label,summary)
        print(scores)
        break



    # df = pd.read_csv('/data/home1/arunsg/dlnlp_report/Text-Summarization-using-T5/news_summary.csv',encoding='latin')
    # df = df.dropna()

    for i in range(val_data.num_rows):
        
        # print(i)
        # if(i==20):
            # break

        text,label = val_data[i]['article'],val_data[i]['highlights']
        
        # print(text)
        # print(label)
        # print(type(text))

        # pred =  get_summary(text)
        pred = model(text)
        pred = pred[0]['generated_text']

        
        # print(pred)
        
        scores = scorer.score(label, pred)
        # print(scores)
        # break
        t5_rouge1Score[0] +=scores['rouge1'].precision
        t5_rouge1Score[1] +=scores['rouge1'].recall
        t5_rouge1Score[2] +=scores['rouge1'].fmeasure

        t5_rouge2Score[0] +=scores['rouge2'].precision
        t5_rouge2Score[1] +=scores['rouge2'].recall
        t5_rouge2Score[2] +=scores['rouge2'].fmeasure

        t5_rougeLScore[0] +=scores['rougeL'].precision
        t5_rougeLScore[1] +=scores['rougeL'].recall
        t5_rougeLScore[2] +=scores['rougeL'].fmeasure


    for i in range(3):
        for j in range(3):

            t5_rouge1Score[i][j] = t5_rouge1Score[i][j]

    return (t5_rouge1Score,t5_rouge2Score,t5_rougeLScore)