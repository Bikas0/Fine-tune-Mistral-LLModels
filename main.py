# import torch 
# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# # device = "cuda" # the device to load the model onto
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model = AutoModelForCausalLM.from_pretrained(
#     "mistralai/Mistral-7B-Instruct-v0.2",
#     torch_dtype=torch.bfloat16,
#     quantization_config=BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_compute_dtype=torch.bfloat16,
#         bnb_4bit_quant=False,
#         bnb_4bit_type = 'nf4',

#     ),
#     ).to(device)
# tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2").to(device)

# text = """
# বন্তন্তরাধীন জমির নূন্যপক্ষে ২৫ বছরের মালিকানার ধারাবাহিক বিবরণ (যথাযথ ক্ষেত্রে ওয়ারিশ ও বায়া দলিল
# সমূহের বিস্তারিত বিবরণ) এবং হস্তান্তরের উদ্দেশ্য, সম্পত্তির দখল ইজমেন্ট স্বত্ব এবং হস্তান্তর সম্পর্কিত উল্লেখযোগ্য
# মন্তব্য (যদি থাকে) সম্পর্কিত বিবরণ $

# জেলা রাজবাড়ী থানা রাজবাড়ী সদর অন্তত সাবেক ১৬ নং হাল ১০৫ নং মৌজা নয়নদিয়া মধ্যে জমার জনমি গত
# ইংরাজী ২০/০৮/১৯৯৫ তারিখের ৬০৬৩ নং কবলা দলিল মূলে মোঃ আমিনুল ইসলাম খরিদ করি স্ব্কবান 'ও
# দখলকার থাকা অবস্থায় উক্ত মোঃ আমিনুল ইসলাম এয় নিকট হইতে আমি গত ইংরাজী ২১/০৫/২০০৫ তারিখের
# ৩৬১৫ নং কবলা দলিল মূলে খরিদ করি ও গত ইংরাজী ১২/০৪/১৯৯৬ তারিখের ২২০৬ নং কবলা দলিল মূলে প্রীতিশ
# কুমার সরকার খরিদ করিয়া স্বভুবান ও দখলকার থাকা অবস্থায় উদ্ত পরীতিশ কুমার সরকার এর নিকট হইতে আমি গত
# ইংরাজী ২৭/০৫/২০০২ তারিখের ৩৫৩৬ নং কবলা দলিল মূলে খরিদ করি ও গত ইংরাজী ২৪/১২/১৯৮২ তারিখের
# ৮০৩৯ নং কৰলা দলিল মূলে আলহান্ত জলী মন্ডল খরিদ করিয়া স্বভবান ও দখলকার থাকা অবস্থায় উক্ত আলহা্
# আলী মন্ডল এর নিকট হইতে আমি গত ইংরাজী ০৭/১২/২০০৪ তারিখের ৭৯১৭ নং কবলা দলিল মূলে খরিদ করিয়া
# গত ইংরাজী ২৬/০৮/২০০৯ তারিখের A/C (Land) IX-P-I-ov/o»-১০, T.O-IX-P-I-o৯9/o৯-১০ নং
# কেলে আমার নিজ নামে নামজারী, জমাভাগ ও নামপত্তন করিয়া প্রস্তাবিত ৭০/১ নং খতিয়ান খুলিয়া স্বড়বান ও
# দগখলকার জাছি। এইক্ষণ আমি নিয়নলিখিত দাগ হইতে ১১ (১) নং তপশিলে ৩৪ শতাংশ জমি তোমাদের বরাবর হেবা
# ঘোবণা পত্র করিয়া দিলাম । ১১ (২) নং তপশিনে সাবেক ১৫ নং হাল ৪০ নং মৌজা নাওডুৰি মধ্যে জমার জনমি গত
# ইংাজী ০৭/১২/১৯৮৬ তারিখের ৬৩১০ নং কবলা দলিল মূলে জাহাঙ্গীর হোসেন গং খরিদ করিয়া ্রকবান ও দখলকার
# থাকা অবস্থায় উক্ত জাহাঙ্গীর হোসেন গং এর নিকট হইতে আমি গত ইংরাজী ১৫/১১/১৯৯৯ তারিখের ৭৬৯০ নং
# কবলা দলিল মূলে খরিদ করিয়া গত ইংরাজী ২৬/০৮/২০০৯ তারিখের A/C (Land) DC-P-[-৩৭/o৯-১০, T.0-
# DX P-1-৩৯৬/০৯-১০ নং কেসে আমার নিজ নামে নামজারী, জমাভাগ ও নামপ্তন করাইয়া প্রস্তাবিত ৪৩৮ নং
# খতিয়ান খুলিয়া স্বান ও দখলকার আছি। এইক্ষণ আমি নিয়লিখিত দাগ হইতে ১১ (২) নং তপশিলে ২৩.৫০
# শতাংশ সর্বমোট ৫৭.৫০ শতাংশ জমি তোমাদের বরাবর হেবা ঘোষণা পর্ন করিয়া দিলাম

# তুমি ১নং দলিল গ্রহীতা জমার ভরবজ্গাত পুত সন্তান ও তুমি ২নং দলিল গ্রহীতা আমার বিবাহিতা স্ত্রী হইতেছ।

# """
# # prompt = "Give me a short introduction to large language model."
# prompt = f"""
#     Permorm the following actions:
#     1 - Extract all real Person Names
#     2 - Find out Date only 
#         example:
#             1. ২০/০৮/১৯৯৫
#             2. ২১/০৫/২০০৫
#     3 - Extract all exact Locations
#     4 - Extract all Numbers
#     5 - Extract all email id
#         example:
#             bikasictiu1718@gmail.com
#     6 - Extract all Mobile Numbers
#         example:
#             1. +8801997515363
#             2. 01326282712
#     6 - Output a json object that contains the following keys: Person Names, Dates, Mobile Numbers, Locations, email, and Numbers.
#     Separate your answers with line breaks.
#     Text:
#     ```{text}```
#     """
# messages = [
#     {"role": "user", "content": "You are a helpful assistant."},
#     {"role": "assistant", "content": "Well, I'm quite partial to a good similarity keyword extraction tools for Bengali Text"},
#     {"role": "user", "content": prompt}
# ]

# encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

# model_inputs = encodeds.to(device)
# model.to(device)

# generated_ids = model.generate(encodeds, max_new_tokens=1000, do_sample=True)
# decoded = tokenizer.batch_decode(generated_ids)
# print(decoded[0])

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# device = "cuda" # the device to load the model onto
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    torch_dtype=torch.bfloat16,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant=False,
        bnb_4bit_type = 'nf4',

    ),
    )
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

text = """

বন্তন্তরাধীন জমির নূন্যপক্ষে ২৫ বছরের মালিকানার ধারাবাহিক বিবরণ (যথাযথ ক্ষেত্রে ওয়ারিশ ও বায়া দলিল
সমূহের বিস্তারিত বিবরণ) এবং হস্তান্তরের উদ্দেশ্য, ba-systems@yahoo.com সম্পত্তির দখল ইজমেন্ট স্বত্ব এবং হস্তান্তর সম্পর্কিত উল্লেখযোগ্য
মন্তব্য (যদি থাকে) সম্পর্কিত বিবরণ $

জেলা রাজবাড়ী থানা রাজবাড়ী সদর অন্তত সাবেক ১৬ নং হাল ১০৫ নং মৌজা নয়নদিয়া মধ্যে জমার জনমি গত
ইংরাজী ২০/০৮/১৯৯৫ তারিখের ৬০৬৩ নং কবলা দলিল মূলে মোঃ আমিনুল ইসলাম খরিদ করি স্ব্কবান 'ও
দখলকার থাকা অবস্থায় উক্ত মোঃ আমিনুল ইসলাম এয় নিকট হইতে আমি গত ইংরাজী ২১/০৫/২০০৫ তারিখের
৩৬১৫ নং কবলা দলিল মূলে খরিদ করি ও গত ইংরাজী ১২/০৪/১৯৯৬ তারিখের ২২০৬ নং কবলা দলিল মূলে প্রীতিশ
কুমার সরকার খরিদ করিয়া স্বভুবান ও দখলকার থাকা অবস্থায় উদ্ত পরীতিশ কুমার সরকার এর নিকট হইতে আমি গত
ইংরাজী ২৭/০৫/২০০২ তারিখের ৩৫৩৬ নং কবলা দলিল মূলে খরিদ করি ও গত ইংরাজী ২৪/১২/১৯৮২ তারিখের
৮০৩৯ নং কৰলা দলিল মূলে আলহান্ত জলী মন্ডল খরিদ করিয়া স্বভবান ও দখলকার থাকা অবস্থায় উক্ত আলহা্
আলী মন্ডল এর নিকট হইতে আমি গত ইংরাজী ০৭/১২/২০০৪ তারিখের ৭৯১৭ নং কবলা দলিল মূলে খরিদ করিয়া
গত ইংরাজী ২৬/০৮/২০০৯ তারিখের A/C (Land) IX-P-I-ov/o»-১০, T.O-IX-P-I-o৯9/o৯-১০ নং
কেলে আমার নিজ নামে নামজারী, systems@org.com জমাভাগ ও নামপত্তন করিয়া প্রস্তাবিত ৭০/১ নং খতিয়ান খুলিয়া স্বড়বান ও
দগখলকার জাছি। এইক্ষণ আমি নিয়নলিখিত দাগ হইতে ১১ (১) নং তপশিলে ৩৪ শতাংশ জমি তোমাদের বরাবর হেবা
ঘোবণা পত্র করিয়া দিলাম । ১১ (২) নং তপশিনে সাবেক ১৫ নং হাল ৪০ নং মৌজা নাওডুৰি মধ্যে জমার জনমি গত
ইংাজী ০৭/১২/১৯৮৬ তারিখের ৬৩১০ নং কবলা দলিল মূলে জাহাঙ্গীর হোসেন গং খরিদ করিয়া ্রকবান ও দখলকার
থাকা অবস্থায় উক্ত জাহাঙ্গীর হোসেন গং এর নিকট হইতে আমি গত ইংরাজী ১৫/১১/১৯৯৯ তারিখের ৭৬৯০ নং
কবলা দলিল মূলে খরিদ করিয়া গত ইংরাজী ২৬/০৮/২০০৯ তারিখের A/C (Land) DC-P-[-৩৭/o৯-১০, T.0-
DX P-1-৩৯৬/০৯-১০ নং কেসে আমার নিজ নামে নামজারী, জমাভাগ ও নামপ্তন করাইয়া প্রস্তাবিত ৪৩৮ নং
খতিয়ান খুলিয়া স্বান ও দখলকার আছি। এইক্ষণ আমি নিয়লিখিত দাগ হইতে ১১ (২) নং তপশিলে ২৩.৫০
শতাংশ সর্বমোট ৫৭.৫০ শতাংশ জমি তোমাদের বরাবর হেবা ঘোষণা পর্ন করিয়া দিলাম +8801369282712

তুমি ১নং দলিল গ্রহীতা জমার ভরবজ্গাত পুত সন্তান ও তুমি ২নং দলিল গ্রহীতা আমার বিবাহিতা স্ত্রী হইতেছ।

"""

prompt = f"""
    Permorm the following actions:
    1 - Extract all real Person Names
    2 - Find out Date only 
        example:
            1. ২০/০৮/১৯৯৫
            2. ২১/০৫/২০০৫
    3 - Extract exact Locations
    4 - Extract Land buy and sell Price
    5 - Extract all email id
        example:
            bikasictiu1718@gmail.com
    6 - Extract all Mobile Numbers
        example:
            1. +8801997515363
            2. 01326282712
    7 - Find out Khatian Number with information from the text
    8 - Output a json object that contains the following keys: Person Names, Dates, Mobile Numbers, Locations, Email, Khatian and LandBuy and Sell.
    Separate your answers with line breaks.
    Text:
    ```{text}```
    """

messages = [
    {"role": "user", "content": "You are a helpful assistant. You are a keyword extractor."},
    {"role": "assistant", "content": "Well, I'm quite partial to a good keyword extraction tools for Bengali Text"},
    {"role": "user", "content": prompt}
]

encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

model_inputs = encodeds.to(device)
# model.to(device)

# Define generation parameters
# generation_params = {
#     "max_new_tokens": 1000,
#     "do_sample": True,
#     "temperature": 0.001,  # Adjust the temperature here
#     "top_p": 0.0,        # Adjust the top_p here
# }



# Generate text
# generated_ids = model.generate(encodeds, **generation_params)
# decoded = tokenizer.batch_decode(generated_ids)
# print(decoded[0])


generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True, temperature=0.001, top_p= 0.0)
decoded = tokenizer.batch_decode(generated_ids)
print(decoded[0])