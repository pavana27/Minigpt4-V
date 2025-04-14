# Combined Evaluation Modules
from dotenv import load_dotenv,find_dotenv
from multiprocessing.pool import Pool
import argparse
import ast
import json
import openai
import os
import time



# ===== Module 1: Evaluation Logic =====
def run_module_1(args):


    def parse_args():
        parser = argparse.ArgumentParser(description="question-answer-generation-using-gpt-3")
        parser.add_argument("--pred_path", required=True, help="The path to file containing prediction.")
        parser.add_argument("--output_dir", required=True, help="The path to save annotation json files.")
        parser.add_argument("--output_json", required=True, help="The path to save annotation final combined json file.")
        parser.add_argument("--api_key", required=True, help="OpenAI API key.")
        parser.add_argument("--num_tasks", required=True, type=int, help="Number of splits.")
        args = parser.parse_args()
        return args


    def annotate(prediction_set, caption_files, output_dir):
        """
        Evaluates question and answer pairs using GPT-3
        Returns a score for correctness.
        """
        for file in caption_files:
            key = file[:-5] # Strip file extension
            qa_set = prediction_set[key]
            question = qa_set['q']
            answer = qa_set['a']
            pred = qa_set['pred']
            try:
                completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system",
                            "content": 
                                "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
                                "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"
                                "------"
                                "##INSTRUCTIONS: "
                                "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
                                "- Consider synonyms or paraphrases as valid matches.\n"
                                "- Evaluate the correctness of the prediction compared to the answer."
                        },
                        {
                            "role": "user",
                            "content":
                                "Please evaluate the following video-based question-answer pair:\n\n"
                                f"Question: {question}\n"
                                f"Correct Answer: {answer}\n"
                                f"Predicted Answer: {pred}\n\n"
                                "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. "
                                "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is  a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING."
                                "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                                "For example, your response should look like this: {'pred': 'yes', 'score': 4.8}."
                        }
                    ]
                )
                response_message = completion["choices"][0]["message"]["content"]
                response_dict = ast.literal_eval(response_message)
                result_qa_pair = [response_dict, qa_set]

                with open(f"{output_dir}/{key}.json", "w") as f:
                    json.dump(result_qa_pair, f)

            except Exception as e:
                print(f"Error processing file '{key}': {e}")


    def main():
        """
        Main function to control the flow of the program.
        """
        args = parse_args()

        file = open(args.pred_path)
        pred_contents = json.load(file)

        video_id_counts = {}
        new_pred_contents = []

        for sample in pred_contents:
            video_id = sample['video_name']
            if video_id in video_id_counts:
                video_id_counts[video_id] += 1
            else:
                video_id_counts[video_id] = 0

            new_sample = sample
            new_sample['video_name'] = f"{video_id}_{video_id_counts[video_id]}"
            new_pred_contents.append(new_sample)

        id_list = [x['video_name'] for x in new_pred_contents]
        caption_files = [f"{id}.json" for id in id_list]

        output_dir = args.output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        prediction_set = {}
        for sample in new_pred_contents:
            id = sample['video_name']
            question = sample['Q']
            answer = sample['A']
            pred = sample['pred']
            qa_set = {"q": question, "a": answer, "pred": pred}
            prediction_set[id] = qa_set

        openai.api_key = args.api_key
        num_tasks = args.num_tasks

        while True:
            try:
                completed_files = os.listdir(output_dir)
                print(f"completed_files: {len(completed_files)}")

                incomplete_files = [f for f in caption_files if f not in completed_files]
                print(f"incomplete_files: {len(incomplete_files)}")

                if len(incomplete_files) == 0:
                    break
                if len(incomplete_files) <= num_tasks:
                    num_tasks = 1

                part_len = len(incomplete_files) // num_tasks
                all_parts = [incomplete_files[i:i + part_len] for i in range(0, len(incomplete_files), part_len)]
                task_args = [(prediction_set, part, args.output_dir) for part in all_parts]

                with Pool() as pool:
                    pool.starmap(annotate, task_args)

            except Exception as e:
                print(f"Error: {e}")

        combined_contents = {}
        json_path = args.output_json

        for file_name in os.listdir(output_dir):
            if file_name.endswith(".json"):
                file_path = os.path.join(output_dir, file_name)
                with open(file_path, "r") as json_file:
                    content = json.load(json_file)
                    combined_contents[file_name[:-5]] = content

        with open(json_path, "w") as json_file:
            json.dump(combined_contents, json_file)
        print("All evaluation completed!")

        score_sum = 0
        count = 0
        yes_count = 0
        no_count = 0
        for key, result in combined_contents.items():
            count += 1
            try :
                score_match = result[0]['score']
                score = int(score_match)
                score_sum += score
            except:
                print("Score not found for", key)
                continue

            try:
                pred = result[0]['pred']
                if "yes" in pred.lower():
                    yes_count += 1
                elif "no" in pred.lower():
                    no_count += 1
            except:
                print("Prediction not found for", key)
                continue

        average_score = score_sum / count
        accuracy = yes_count / (yes_count + no_count)
        print("Yes count:", yes_count)
        print("No count:", no_count)
        print("Accuracy:", accuracy)
        print("Average score:", average_score)


        main()




# ===== Module 2: Evaluation Logic =====
def run_module_2(args):

    def parse_args():

        parser = argparse.ArgumentParser(description="question-answer-generation-using-gpt-3")
        parser.add_argument("--pred_path", required=True, help="The path to file containing prediction.")
        parser.add_argument("--output_dir", required=True, help="The path to save annotation json files.")
        parser.add_argument("--output_json", required=True, help="The path to save annotation final combined json file.")
        parser.add_argument("--api_key", required=True, help="OpenAI API key.")
        parser.add_argument("--num_tasks", required=True, type=int, help="Number of splits.")
        args = parser.parse_args()
        return args


    def annotate(prediction_set, caption_files, output_dir):
        """
        Evaluates question and answer pairs using GPT-3
        Returns a score for correctness.
        """
        for file in caption_files:
            key = file[:-5] # Strip file extension
            qa_set = prediction_set[key]
            question = qa_set['q']
            answer = qa_set['a']
            pred = qa_set['pred']
            try:
                completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system",
                            "content": 
                                "You are an intelligent chatbot designed for evaluating the factual accuracy of generative outputs for video-based question-answer pairs. "
                                "Your task is to compare the predicted answer with the correct answer and determine if they are factually consistent. Here's how you can accomplish the task:"
                                "------"
                                "##INSTRUCTIONS: "
                                "- Focus on the factual consistency between the predicted answer and the correct answer. The predicted answer should not contain any misinterpretations or misinformation.\n"
                                "- The predicted answer must be factually accurate and align with the video content.\n"
                                "- Consider synonyms or paraphrases as valid matches.\n"
                                "- Evaluate the factual accuracy of the prediction compared to the answer."
                        },
                        {
                            "role": "user",
                            "content":
                                "Please evaluate the following video-based question-answer pair:\n\n"
                                f"Question: {question}\n"
                                f"Correct Answer: {answer}\n"
                                f"Predicted Answer: {pred}\n\n"
                                "Provide your evaluation only as a factual accuracy score where the factual accuracy score is an integer value between 0 and 5, with 5 indicating the highest level of factual consistency. "
                                "Please generate the response in the form of a Python dictionary string with keys 'score', where its value is the factual accuracy score in INTEGER, not STRING."
                                "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                                "For example, your response should look like this: {''score': 4.8}."
                        }
                    ]
                )
                response_message = completion.choices[0].message.content
                response_dict = ast.literal_eval(response_message)
                result_qa_pair = [response_dict, qa_set]

                with open(f"{output_dir}/{key}.json", "w") as f:
                    json.dump(result_qa_pair, f)

            except Exception as e:
                print(f"Error processing file '{key}': {e}")
                time.sleep(2)

    def main():
        """
        Main function to control the flow of the program.
        """
        args = parse_args()

        file = open(args.pred_path)
        pred_contents = json.load(file)
        print(args.pred_path,len(pred_contents))
        video_id_counts = {}
        new_pred_contents = []

        for sample in pred_contents:
            video_id = sample['video_id'] if 'video_id' in sample else sample['video_name']
            if video_id in video_id_counts:
                video_id_counts[video_id] += 1
            else:
                video_id_counts[video_id] = 0

            new_sample = sample
            new_sample['video_name'] = f"{video_id}_{video_id_counts[video_id]}"
            new_pred_contents.append(new_sample)

        id_list = [x['video_name'] for x in new_pred_contents]
        caption_files = [f"{id}.json" for id in id_list]

        output_dir = args.output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        prediction_set = {}
        for sample in new_pred_contents:
            id = sample['video_name']
            question = sample['Q']
            answer = sample['A']
            pred = sample['pred']
            qa_set = {"q": question, "a": answer, "pred": pred}
            prediction_set[id] = qa_set

        openai.api_key = args.api_key
        num_tasks = args.num_tasks

        while True:
            try:
                completed_files = os.listdir(output_dir)
                print(f"completed_files: {len(completed_files)}")

                incomplete_files = [f for f in caption_files if f not in completed_files]
                print(f"incomplete_files: {len(incomplete_files)}")

                if len(incomplete_files) == 0:
                    break
                if len(incomplete_files) <= num_tasks:
                    num_tasks = 1

                part_len = len(incomplete_files) // num_tasks
                all_parts = [incomplete_files[i:i + part_len] for i in range(0, len(incomplete_files), part_len)]
                task_args = [(prediction_set, part, args.output_dir) for part in all_parts]

                with Pool() as pool:
                    pool.starmap(annotate, task_args)

            except Exception as e:
                print(f"Error: {e}")

        combined_contents = {}
        json_path = args.output_json

        for file_name in os.listdir(output_dir):
            if file_name.endswith(".json"):
                file_path = os.path.join(output_dir, file_name)
                with open(file_path, "r") as json_file:
                    content = json.load(json_file)
                    combined_contents[file_name[:-5]] = content

        with open(json_path, "w") as json_file:
            json.dump(combined_contents, json_file)
        print("All evaluation completed!")

        score_sum = 0
        count = 0
        for key, result in combined_contents.items():
            count += 1
            score_match = result[0]['score']
            score = int(score_match)
            score_sum += score
        average_score = score_sum / count

        print("Average score for correctness:", average_score)



# ===== Module 3: Evaluation Logic =====
def run_module_3(args):

    def parse_args():
        parser = argparse.ArgumentParser(description="question-answer-generation-using-gpt-3")
        parser.add_argument("--pred_path", required=True, help="The path to file containing prediction.")
        parser.add_argument("--output_dir", required=True, help="The path to save annotation json files.")
        parser.add_argument("--output_json", required=True, help="The path to save annotation final combined json file.")
        parser.add_argument("--api_key", required=True, help="OpenAI API key.")
        parser.add_argument("--num_tasks", required=True, type=int, help="Number of splits.")
        args = parser.parse_args()
        return args


    def annotate(prediction_set, caption_files, output_dir):
        """
        Evaluates question and answer pairs using GPT-3 and
        returns a score for detailed orientation.
        """
        for file in caption_files:
            key = file[:-5] # Strip file extension
            qa_set = prediction_set[key]
            question = qa_set['q']
            answer = qa_set['a']
            pred = qa_set['pred']
            try:
                completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system",
                            "content":
                                "You are an intelligent chatbot designed for evaluating the detail orientation of generative outputs for video-based question-answer pairs. "
                                "Your task is to compare the predicted answer with the correct answer and determine its level of detail, considering both completeness and specificity. Here's how you can accomplish the task:"
                                "------"
                                "##INSTRUCTIONS: "
                                "- Check if the predicted answer covers all major points from the video. The response should not leave out any key aspects.\n"
                                "- Evaluate whether the predicted answer includes specific details rather than just generic points. It should provide comprehensive information that is tied to specific elements of the video.\n"
                                "- Consider synonyms or paraphrases as valid matches.\n"
                                "- Provide a single evaluation score that reflects the level of detail orientation of the prediction, considering both completeness and specificity."
                        },
                        {
                            "role": "user",
                            "content":
                                "Please evaluate the following video-based question-answer pair:\n\n"
                                f"Question: {question}\n"
                                f"Correct Answer: {answer}\n"
                                f"Predicted Answer: {pred}\n\n"
                                "Provide your evaluation only as a detail orientation score where the detail orientation score is an integer value between 0 and 5, with 5 indicating the highest level of detail orientation. "
                                "Please generate the response in the form of a Python dictionary string with keys 'score', where its value is the detail orientation score in INTEGER, not STRING."
                                "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                                "For example, your response should look like this: {''score': 4.8}."
                        }
                    ]
                )
                response_message = completion.choices[0].message.content
                response_dict = ast.literal_eval(response_message)
                result_qa_pair = [response_dict, qa_set]

                with open(f"{output_dir}/{key}.json", "w") as f:
                    json.dump(result_qa_pair, f)

            except Exception as e:
                print(f"Error processing file '{key}': {e}")
                time.sleep(2)

    def main():
        """
        Main function to control the flow of the program.
        """
        args = parse_args()

        file = open(args.pred_path)
        pred_contents = json.load(file)

        video_id_counts = {}
        new_pred_contents = []

        for sample in pred_contents:
            video_id = sample['video_name']
            if video_id in video_id_counts:
                video_id_counts[video_id] += 1
            else:
                video_id_counts[video_id] = 0

            new_sample = sample
            new_sample['video_name'] = f"{video_id}_{video_id_counts[video_id]}"
            new_pred_contents.append(new_sample)

        id_list = [x['video_name'] for x in new_pred_contents]
        caption_files = [f"{id}.json" for id in id_list]

        output_dir = args.output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        prediction_set = {}
        for sample in new_pred_contents:
            id = sample['video_id'] if 'video_id' in sample else sample['video_name']
            question = sample['Q']
            answer = sample['A']
            pred = sample['pred']
            qa_set = {"q": question, "a": answer, "pred": pred}
            prediction_set[id] = qa_set

        openai.api_key = args.api_key
        num_tasks = args.num_tasks


        while True:
            try:
                completed_files = os.listdir(output_dir)
                print(f"completed_files: {len(completed_files)}")

                incomplete_files = [f for f in caption_files if f not in completed_files]
                print(f"incomplete_files: {len(incomplete_files)}")

                if len(incomplete_files) == 0:
                    break
                if len(incomplete_files) <= num_tasks:
                    num_tasks = 1

                part_len = len(incomplete_files) // num_tasks
                all_parts = [incomplete_files[i:i + part_len] for i in range(0, len(incomplete_files), part_len)]
                task_args = [(prediction_set, part, args.output_dir) for part in all_parts]

                with Pool() as pool:
                    pool.starmap(annotate, task_args)

            except Exception as e:
                print(f"Error: {e}")

        combined_contents = {}
        json_path = args.output_json

        for file_name in os.listdir(output_dir):
            if file_name.endswith(".json"):
                file_path = os.path.join(output_dir, file_name)
                with open(file_path, "r") as json_file:
                    content = json.load(json_file)
                    combined_contents[file_name[:-5]] = content

        with open(json_path, "w") as json_file:
            json.dump(combined_contents, json_file)
        print("All evaluation completed!")

        score_sum = 0
        count = 0
        for key, result in combined_contents.items():
            count += 1
            score_match = result[0]['score']
            score = int(score_match)
            score_sum += score
        average_score = score_sum / count

        print("Average score for detailed orientation:", average_score)


        main()



# ===== Module 4: Evaluation Logic =====
def run_module_4(args):

    def parse_args():
        parser = argparse.ArgumentParser(description="question-answer-generation-using-gpt-3")
        parser.add_argument("--pred_path", required=True, help="The path to file containing prediction.")
        parser.add_argument("--output_dir", required=True, help="The path to save annotation json files.")
        parser.add_argument("--output_json", required=True, help="The path to save annotation final combined json file.")
        parser.add_argument("--api_key", required=True, help="OpenAI API key.")
        parser.add_argument("--num_tasks", required=True, type=int, help="Number of splits.")
        args = parser.parse_args()
        return args


    def annotate(prediction_set, caption_files, output_dir):
        """
        Evaluates question and answer pairs using GPT-3 and
        returns a score for contextual understanding.
        """
        for file in caption_files:
            key = file[:-5] # Strip file extension
            qa_set = prediction_set[key]
            question = qa_set['q']
            answer = qa_set['a']
            pred = qa_set['pred']
            try:
                completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system",
                            "content":
                                "You are an intelligent chatbot designed for evaluating the contextual understanding of generative outputs for video-based question-answer pairs. "
                                "Your task is to compare the predicted answer with the correct answer and determine if the generated response aligns with the overall context of the video content. Here's how you can accomplish the task:"
                                "------"
                                "##INSTRUCTIONS: "
                                "- Evaluate whether the predicted answer aligns with the overall context of the video content. It should not provide information that is out of context or misaligned.\n"
                                "- The predicted answer must capture the main themes and sentiments of the video.\n"
                                "- Consider synonyms or paraphrases as valid matches.\n"
                                "- Provide your evaluation of the contextual understanding of the prediction compared to the answer."
                        },
                        {
                            "role": "user",
                            "content":
                                "Please evaluate the following video-based question-answer pair:\n\n"
                                f"Question: {question}\n"
                                f"Correct Answer: {answer}\n"
                                f"Predicted Answer: {pred}\n\n"
                                "Provide your evaluation only as a contextual understanding score where the contextual understanding score is an integer value between 0 and 5, with 5 indicating the highest level of contextual understanding. "
                                "Please generate the response in the form of a Python dictionary string with keys 'score', where its value is contextual understanding score in INTEGER, not STRING."
                                "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                                "For example, your response should look like this: {''score': 4.8}."
                        }
                    ]
                )
                response_message = completion.choices[0].message.content
                response_dict = ast.literal_eval(response_message)
                result_qa_pair = [response_dict, qa_set]

                with open(f"{output_dir}/{key}.json", "w") as f:
                    json.dump(result_qa_pair, f)

            except Exception as e:
                print(f"Error processing file '{key}': {e}")
                time.sleep(2)

    def main():
        """
        Main function to control the flow of the program.
        """
        args = parse_args()

        file = open(args.pred_path)
        pred_contents = json.load(file)

        video_id_counts = {}
        new_pred_contents = []

        for sample in pred_contents:
            video_id = sample['video_name']
            if video_id in video_id_counts:
                video_id_counts[video_id] += 1
            else:
                video_id_counts[video_id] = 0

            new_sample = sample
            new_sample['video_name'] = f"{video_id}_{video_id_counts[video_id]}"
            new_pred_contents.append(new_sample)

        id_list = [x['video_name'] for x in new_pred_contents]
        caption_files = [f"{id}.json" for id in id_list]

        output_dir = args.output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        prediction_set = {}
        for sample in new_pred_contents:
            id = sample['video_name'] if 'video_name' in sample else sample['video_id']
            question = sample['Q']
            answer = sample['A']
            pred = sample['pred']
            qa_set = {"q": question, "a": answer, "pred": pred}
            prediction_set[id] = qa_set

        openai.api_key = args.api_key
        num_tasks = args.num_tasks
        while True:
            try:
                completed_files = os.listdir(output_dir)
                print(f"completed_files: {len(completed_files)}")

                incomplete_files = [f for f in caption_files if f not in completed_files]
                print(f"incomplete_files: {len(incomplete_files)}")

                if len(incomplete_files) == 0:
                    break
                if len(incomplete_files) <= num_tasks:
                    num_tasks = 1

                part_len = len(incomplete_files) // num_tasks
                all_parts = [incomplete_files[i:i + part_len] for i in range(0, len(incomplete_files), part_len)]
                task_args = [(prediction_set, part, args.output_dir) for part in all_parts]

                with Pool() as pool:
                    pool.starmap(annotate, task_args)

            except Exception as e:
                print(f"Error: {e}")

        combined_contents = {}
        json_path = args.output_json

        for file_name in os.listdir(output_dir):
            if file_name.endswith(".json"):
                file_path = os.path.join(output_dir, file_name)
                with open(file_path, "r") as json_file:
                    content = json.load(json_file)
                    combined_contents[file_name[:-5]] = content

        with open(json_path, "w") as json_file:
            json.dump(combined_contents, json_file)
        print("All evaluation completed!")

        score_sum = 0
        count = 0
        for key, result in combined_contents.items():
            count += 1
            score_match = result[0]['score']
            score = int(score_match)
            score_sum += score
        average_score = score_sum / count

        print("Average score for contextual understanding:", average_score)


        main()



# ===== Module 5: Evaluation Logic =====
def run_module_5(args):



    def parse_args():
        parser = argparse.ArgumentParser(description="question-answer-generation-using-gpt-3")
        parser.add_argument("--pred_path", required=True, help="The path to file containing prediction.")
        parser.add_argument("--output_dir", required=True, help="The path to save annotation json files.")
        parser.add_argument("--output_json", required=True, help="The path to save annotation final combined json file.")
        parser.add_argument("--api_key", required=True, help="OpenAI API key.")
        parser.add_argument("--num_tasks", required=True, type=int, help="Number of splits.")
        args = parser.parse_args()
        return args


    def annotate(prediction_set, caption_files, output_dir):
        """
        Evaluates question and answer pairs using GPT-3 and
        returns a score for temporal understanding.
        """
        for file in caption_files:
            key = file[:-5] # Strip file extension
            qa_set = prediction_set[key]
            question = qa_set['q']
            answer = qa_set['a']
            pred = qa_set['pred']
            try:
                completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system",
                            "content":
                                "You are an intelligent chatbot designed for evaluating the temporal understanding of generative outputs for video-based question-answer pairs. "
                                "Your task is to compare the predicted answer with the correct answer and determine if they correctly reflect the temporal sequence of events in the video content. Here's how you can accomplish the task:"
                                "------"
                                "##INSTRUCTIONS: "
                                "- Focus on the temporal consistency between the predicted answer and the correct answer. The predicted answer should correctly reflect the sequence of events or details as they are presented in the video content.\n"
                                "- Consider synonyms or paraphrases as valid matches, but only if the temporal order is maintained.\n"
                                "- Evaluate the temporal accuracy of the prediction compared to the answer."
                        },
                        {
                            "role": "user",
                            "content":
                                "Please evaluate the following video-based question-answer pair:\n\n"
                                f"Question: {question}\n"
                                f"Correct Answer: {answer}\n"
                                f"Predicted Answer: {pred}\n\n"
                                "Provide your evaluation only as a temporal accuracy score where the temporal accuracy score is an integer value between 0 and 5, with 5 indicating the highest level of temporal consistency. "
                                "Please generate the response in the form of a Python dictionary string with keys 'score', where its value is the temporal accuracy score in INTEGER, not STRING."
                                "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                                "For example, your response should look like this: {''score': 4.8}."
                        }
                    ]
                )
                response_message = completion.choices[0].message.content
                response_dict = ast.literal_eval(response_message)
                result_qa_pair = [response_dict, qa_set]

                with open(f"{output_dir}/{key}.json", "w") as f:
                    json.dump(result_qa_pair, f)

            except Exception as e:
                print(f"Error processing file '{key}': {e}")


    def main():
        """
        Main function to control the flow of the program.
        """
        args = parse_args()

        file = open(args.pred_path)
        pred_contents = json.load(file)

        video_id_counts = {}
        new_pred_contents = []

        for sample in pred_contents:
            video_id = sample['video_name'] if 'video_name' in sample else sample['video_id']
            if video_id in video_id_counts:
                video_id_counts[video_id] += 1
            else:
                video_id_counts[video_id] = 0

            new_sample = sample
            new_sample['video_name'] = f"{video_id}_{video_id_counts[video_id]}"
            new_pred_contents.append(new_sample)

        id_list = [x['video_name'] for x in new_pred_contents]
        caption_files = [f"{id}.json" for id in id_list]

        output_dir = args.output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        prediction_set = {}
        for sample in new_pred_contents:
            id = sample['video_name']
            question = sample['Q']
            answer = sample['A']
            pred = sample['pred']
            qa_set = {"q": question, "a": answer, "pred": pred}
            prediction_set[id] = qa_set

        openai.api_key = args.api_key
        num_tasks = args.num_tasks
        while True:
            try:
                completed_files = os.listdir(output_dir)
                print(f"completed_files: {len(completed_files)}")

                incomplete_files = [f for f in caption_files if f not in completed_files]
                print(f"incomplete_files: {len(incomplete_files)}")

                if len(incomplete_files) == 0:
                    break
                if len(incomplete_files) <= num_tasks:
                    num_tasks = 1

                part_len = len(incomplete_files) // num_tasks
                all_parts = [incomplete_files[i:i + part_len] for i in range(0, len(incomplete_files), part_len)]
                task_args = [(prediction_set, part, args.output_dir) for part in all_parts]

                with Pool() as pool:
                    pool.starmap(annotate, task_args)

            except Exception as e: 
                print(f"Error: {e}")

        combined_contents = {}
        json_path = args.output_json

        for file_name in os.listdir(output_dir):
            if file_name.endswith(".json"):
                file_path = os.path.join(output_dir, file_name)
                with open(file_path, "r") as json_file:
                    content = json.load(json_file)
                    combined_contents[file_name[:-5]] = content

        with open(json_path, "w") as json_file:
            json.dump(combined_contents, json_file)
        print("All evaluation completed!")

        score_sum = 0
        count = 0
        for key, result in combined_contents.items():
            count += 1
            score_match = result[0]['score']
            score = int(score_match)
            score_sum += score
        average_score = score_sum / count

        print("Average score temporal understanding:", average_score)


        main()



# ===== Module 6: Evaluation Logic =====
def run_module_6(args):


    def parse_args():
        parser = argparse.ArgumentParser(description="question-answer-generation-using-gpt-3")
        parser.add_argument("--pred_path", required=True, help="The path to file containing prediction.")
        parser.add_argument("--output_dir", required=True, help="The path to save annotation json files.")
        parser.add_argument("--output_json", required=True, help="The path to save annotation final combined json file.")
        parser.add_argument("--api_key", required=True, help="OpenAI API key.")
        parser.add_argument("--num_tasks", required=True, type=int, help="Number of splits.")
        args = parser.parse_args()
        return args


    def annotate(prediction_set, caption_files, output_dir):
        """
        Evaluates question and answer pairs using GPT-3 and
        returns a score for consistency.
        """
        for file in caption_files:
            key = file[:-5] # Strip file extension
            qa_set = prediction_set[key]
            question1 = qa_set['q1']
            question2 = qa_set['q2']
            answer = qa_set['a']
            pred1 = qa_set['pred1']
            pred2 = qa_set['pred2']
            try:
                completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system",
                            "content":
                                "You are an intelligent chatbot designed for evaluating the consistency of generative outputs for similar video-based question-answer pairs. "
                                "You will be given two very similar questions, a common answer common to both the questions and predicted answers for the two questions ."
                                "Your task is to compare the predicted answers for two very similar question, with a common correct answer and determine if they are consistent. Here's how you can accomplish the task:"
                                "------"
                                "##INSTRUCTIONS: "
                                "- Focus on the consistency between the two predicted answers and the correct answer. Both predicted answers should correspond to the correct answer and to each other, and should not contain any contradictions or significant differences in the conveyed information.\n"
                                "- Both predicted answers must be consistent with each other and the correct answer, in terms of the information they provide about the video content.\n"
                                "- Consider synonyms or paraphrases as valid matches, but only if they maintain the consistency in the conveyed information.\n"
                                "- Evaluate the consistency of the two predicted answers compared to the correct answer."
                        },
                        {
                            "role": "user",
                            "content":
                                "Please evaluate the following video-based question-answer pair:\n\n"
                                f"Question 1: {question1}\n"
                                f"Question 2: {question2}\n"
                                f"Correct Answer: {answer}\n"
                                f"Predicted Answer to Question 1: {pred1}\n"
                                f"Predicted Answer to Question 2: {pred2}\n\n"
                                "Provide your evaluation only as a consistency score where the consistency score is an integer value between 0 and 5, with 5 indicating the highest level of consistency. "
                                "Please generate the response in the form of a Python dictionary string with keys 'score', where its value is the consistency score in INTEGER, not STRING."
                                "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                                "For example, your response should look like this: {''score': 4.8}."
                        }
                    ]
                )
                response_message = completion.choices[0].message.content
                response_dict = ast.literal_eval(response_message)
                result_qa_pair = [response_dict, qa_set]

                with open(f"{output_dir}/{key}.json", "w") as f:
                    json.dump(result_qa_pair, f)

            except Exception as e:
                print(f"Error processing file '{key}': {e}")


    def main():
        """
        Main function to control the flow of the program.
        """
        args = parse_args()

        file = open(args.pred_path)
        pred_contents = json.load(file)

        video_id_counts = {}
        new_pred_contents = []

        for sample in pred_contents:
            video_id = sample['video_name'] if 'video_name' in sample else sample['video_id']
            if video_id in video_id_counts:
                video_id_counts[video_id] += 1
            else:
                video_id_counts[video_id] = 0

            new_sample = sample
            new_sample['video_name'] = f"{video_id}_{video_id_counts[video_id]}"
            new_pred_contents.append(new_sample)

        id_list = [x['video_name'] for x in new_pred_contents]
        caption_files = [f"{id}.json" for id in id_list]

        output_dir = args.output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        prediction_set = {}
        for sample in new_pred_contents:
            id = sample['video_name']
            question1 = sample['Q1']
            question2 = sample['Q1']
            answer = sample['A']
            pred1 = sample['pred1']
            pred2 = sample['pred2']
            qa_set = {"q1": question1, "q2": question2, "a": answer, "pred1": pred1, "pred2": pred2}
            prediction_set[id] = qa_set

        openai.api_key = args.api_key
        num_tasks = args.num_tasks
        while True:
            try:
                completed_files = os.listdir(output_dir)
                print(f"completed_files: {len(completed_files)}")

                incomplete_files = [f for f in caption_files if f not in completed_files]
                print(f"incomplete_files: {len(incomplete_files)}")

                if len(incomplete_files) == 0:
                    break
                if len(incomplete_files) <= num_tasks:
                    num_tasks = 1

                part_len = len(incomplete_files) // num_tasks
                all_parts = [incomplete_files[i:i + part_len] for i in range(0, len(incomplete_files), part_len)]
                task_args = [(prediction_set, part, args.output_dir) for part in all_parts]

                with Pool() as pool:
                    pool.starmap(annotate, task_args)

            except Exception as e: 
                print(f"Error: {e}")

        combined_contents = {}
        json_path = args.output_json

        for file_name in os.listdir(output_dir):
            if file_name.endswith(".json"):
                file_path = os.path.join(output_dir, file_name)
                with open(file_path, "r") as json_file:
                    content = json.load(json_file)
                    combined_contents[file_name[:-5]] = content

        with open(json_path, "w") as json_file:
            json.dump(combined_contents, json_file)
        print("All evaluation completed!")

        score_sum = 0
        count = 0
        for key, result in combined_contents.items():
            count += 1
            score_match = result[0]['score']
            score = int(score_match)
            score_sum += score
        average_score = score_sum / count

        print("Average score for consistency:", average_score)


        main()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Unified evaluation script")
    parser.add_argument("--module", type=int, required=True, help="Module number to run (1-6)")
    parser.add_argument("--pred_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--api_key", required=True)
    parser.add_argument("--num_tasks", type=int, required=True)
    args = parser.parse_args()

    if args.module == 1:
        run_module_1(args)
    elif args.module == 2:
        run_module_2(args)
    elif args.module == 3:
        run_module_3(args)
    elif args.module == 4:
        run_module_4(args)
    elif args.module == 5:
        run_module_5(args)
    elif args.module == 6:
        run_module_6(args)
    else:
        print("Invalid module number. Please choose between 1 and 6.")