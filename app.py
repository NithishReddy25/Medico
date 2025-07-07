import pandas as pd
from flask import Flask, request, jsonify, render_template, session, redirect, url_for, flash
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

# --- Flask App Configuration ---
app = Flask(__name__)
app.secret_key = 'your_very_secret_key_here_change_this_in_production_for_real_security' # IMPORTANT: Change this to a strong, random key!

# --- Customer Care Information ---
CUSTOMER_CARE_PHONE = "+91-9876543210" # Example Indian phone number
CUSTOMER_CARE_EMAIL = "support@healthaid.com" # Example email

# --- In-Memory Storage ---
user_profiles = {}
user_history = {}

# --- HARDCODED DATASET & ML MODEL (No changes here, keeping it concise) ---
all_symptoms = [
    'Fever', 'Cough', 'Sore Throat', 'Headache', 'Fatigue', 'Body Aches', 'Chills',
    'Runny Nose', 'Nasal Congestion', 'Sneezing', 'Nausea', 'Vomiting', 'Diarrhea',
    'Abdominal Pain', 'Constipation', 'Heartburn', 'Indigestion', 'Bloating',
    'Loss of Appetite', 'Dizziness', 'Vertigo', 'Lightheadedness', 'Chest Pain',
    'Shortness of Breath', 'Wheezing', 'Rash', 'Itching', 'Hives', 'Dry Skin',
    'Swelling (Localized)', 'Joint Pain', 'Muscle Pain', 'Stiffness', 'Back Pain',
    'Neck Pain', 'Weakness', 'Numbness', 'Tingling', 'Tremors', 'Difficulty Swallowing',
    'Hoarseness', 'Earache', 'Ringing in Ears (Tinnitus)', 'Vision Blurry',
    'Eye Redness', 'Eye Itchiness', 'Dry Eyes', 'Sensitivity to Light (Photophobia)',
    'Excessive Thirst', 'Frequent Urination', 'Painful Urination', 'Blood in Urine',
    'Discolored Urine', 'Weight Loss (Unexplained)', 'Weight Gain (Unexplained)',
    'Increased Sweating', 'Cold Intolerance', 'Hot Flashes', 'Mood Swings',
    'Irritability', 'Anxiety', 'Depression', 'Difficulty Sleeping (Insomnia)',
    'Excessive Sleepiness', 'Memory Loss', 'Confusion', 'Difficulty Concentrating',
    'Speech Difficulty', 'Balance Problems', 'Jaw Pain', 'Gum Bleeding', 'Toothache',
    'Bad Breath (Halitosis)', 'Mouth Sores', 'Swollen Glands (Lymph Nodes)',
    'Bruising Easily', 'Nosebleed', 'Hair Loss', 'Brittle Nails', 'Leg Cramps',
    'Restless Legs', 'Cold Hands/Feet', 'Varicose Veins', 'Fainting (Syncope)',
    'Seizures', 'Burning Sensation', 'Pressure Sensation', 'Tender Skin',
    'Yellow Skin (Jaundice)', 'Pale Skin', 'Dark Urine', 'Clay-colored Stools',
    'Loss of Smell (Anosmia)', 'Loss of Taste (Ageusia)', 'Metallic Taste',
    'Heart Palpitations', 'Leg Swelling (Edema)', 'Skin Blisters', 'Dry Mouth',
    'Painful Bowel Movements'
]

symptom_disease_raw_data = [
    {'Disease': 'Common Cold', 'Fever': 1, 'Cough': 1, 'Sore Throat': 1, 'Runny Nose': 1, 'Nasal Congestion': 1, 'Sneezing': 1},
    {'Disease': 'Common Cold', 'Fever': 0, 'Cough': 1, 'Sore Throat': 1, 'Headache': 1, 'Runny Nose': 1, 'Nasal Congestion': 1, 'Sneezing': 1},
    {'Disease': 'Flu', 'Fever': 1, 'Cough': 1, 'Sore Throat': 1, 'Headache': 1, 'Fatigue': 1, 'Body Aches': 1, 'Chills': 1},
    {'Disease': 'Flu', 'Fever': 1, 'Cough': 1, 'Fatigue': 1, 'Body Aches': 1, 'Chills': 1, 'Nausea': 1},
    {'Disease': 'Strep Throat', 'Fever': 1, 'Sore Throat': 1, 'Headache': 1, 'Difficulty Swallowing': 1, 'Swollen Glands (Lymph Nodes)': 1},
    {'Disease': 'Strep Throat', 'Fever': 1, 'Sore Throat': 1, 'Rash': 1},
    {'Disease': 'Dengue', 'Fever': 1, 'Headache': 1, 'Body Aches': 1, 'Joint Pain': 1, 'Muscle Pain': 1, 'Rash': 1, 'Nausea': 1},
    {'Disease': 'Dengue', 'Fever': 1, 'Headache': 1, 'Muscle Pain': 1, 'Vomiting': 1, 'Fatigue': 1, 'Loss of Appetite': 1},
    {'Disease': 'Allergies', 'Sneezing': 1, 'Runny Nose': 1, 'Itching': 1, 'Eye Itchiness': 1, 'Nasal Congestion': 1},
    {'Disease': 'Allergies', 'Hives': 1, 'Itching': 1, 'Dry Skin': 1, 'Swelling (Localized)': 1},
    {'Disease': 'Gastroenteritis', 'Nausea': 1, 'Vomiting': 1, 'Diarrhea': 1, 'Abdominal Pain': 1, 'Fever': 1},
    {'Disease': 'Urinary Tract Infection', 'Painful Urination': 1, 'Frequent Urination': 1, 'Abdominal Pain': 1, 'Discolored Urine': 1},
    {'Disease': 'Migraine', 'Headache': 1, 'Nausea': 1, 'Sensitivity to Light (Photophobia)': 1, 'Vision Blurry': 1},
    {'Disease': 'Hypertension', 'Headache': 1, 'Dizziness': 1, 'Vision Blurry': 1, 'Fatigue': 1},
    {'Disease': 'Hypothyroidism', 'Fatigue': 1, 'Weight Gain (Unexplained)': 1, 'Cold Intolerance': 1, 'Dry Skin': 1, 'Hair Loss': 1},
    {'Disease': 'Anemia', 'Fatigue': 1, 'Weakness': 1, 'Pale Skin': 1, 'Shortness of Breath': 1, 'Dizziness': 1},
    {'Disease': 'Anxiety Disorder', 'Anxiety': 1, 'Heart Palpitations': 1, 'Shortness of Breath': 1, 'Tremors': 1, 'Difficulty Sleeping (Insomnia)': 1},
    {'Disease': 'Jaundice', 'Yellow Skin (Jaundice)': 1, 'Dark Urine': 1, 'Clay-colored Stools': 1, 'Itching': 1},
    {'Disease': 'Dehydration', 'Excessive Thirst':1, 'Dry Mouth':1, 'Fatigue':1, 'Dizziness':1, 'Infrequent Urination':1},
    {'Disease': 'Diabetes (Type 2)', 'Excessive Thirst':1, 'Frequent Urination':1, 'Weight Loss (Unexplained)':0, 'Fatigue':1, 'Vision Blurry':1, 'Increased Sweating':0, 'Numbness':1, 'Tingling':1},
    {'Disease': 'Kidney Stones', 'Abdominal Pain':1, 'Back Pain':1, 'Painful Urination':1, 'Blood in Urine':1, 'Nausea':1, 'Vomiting':1},
    {'Disease': 'Pneumonia', 'Fever':1, 'Cough':1, 'Shortness of Breath':1, 'Chest Pain':1, 'Chills':1, 'Fatigue':1, 'Body Aches':1},
    {'Disease': 'Bronchitis', 'Cough':1, 'Sore Throat':1, 'Fatigue':1, 'Wheezing':1, 'Chest Pain':0, 'Fever':0},
    {'Disease': 'Asthma', 'Shortness of Breath':1, 'Wheezing':1, 'Cough':1, 'Chest Pain':0, 'Fatigue':0},
    {'Disease': 'Eczema', 'Rash':1, 'Itching':1, 'Dry Skin':1, 'Swelling (Localized)':0},
    {'Disease': 'Psoriasis', 'Rash':1, 'Itching':1, 'Dry Skin':1, 'Swelling (Localized)':0, 'Joint Pain':0},
    {'Disease': 'Arthritis (Osteo)', 'Joint Pain':1, 'Stiffness':1, 'Swelling (Localized)':0, 'Fatigue':0},
    {'Disease': 'Depression', 'Depression':1, 'Fatigue':1, 'Difficulty Sleeping (Insomnia)':1, 'Loss of Appetite':0, 'Weight Loss (Unexplained)':0, 'Irritability':1, 'Confusion':0, 'Memory Loss':1},
    {'Disease': 'Vertigo (BPPV)', 'Vertigo':1, 'Dizziness':1, 'Nausea':1, 'Vomiting':0, 'Headache':0},
    {'Disease': 'Food Poisoning', 'Nausea':1, 'Vomiting':1, 'Diarrhea':1, 'Abdominal Pain':1, 'Fever':1, 'Chills':1},
]

ml_data_rows = []
for entry in symptom_disease_raw_data:
    row_dict = {symptom: entry.get(symptom, 0) for symptom in all_symptoms}
    row_dict['Disease'] = entry['Disease']
    ml_data_rows.append(row_dict)

symptom_data_for_ml = pd.DataFrame(ml_data_rows)
X = symptom_data_for_ml.drop('Disease', axis=1)
y = symptom_data_for_ml['Disease']
X = X[all_symptoms]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

disease_info_raw_data = [
    {
        'Disease': 'Common Cold',
        'Description': 'A common viral infection of the nose and throat, causing runny nose, sore throat, cough, and congestion.',
        'Precautions': 'Rest, stay hydrated, wash hands frequently, avoid close contact, use tissues when coughing/sneezing.',
        'Medications': 'Pain relievers (e.g., paracetamol, ibuprofen), decongestants, cough suppressants (e.g., dextromethorphan, guaifenesin), saline nasal sprays.',
        'Diets': 'Warm liquids (tea, broth), chicken soup, vitamin C rich fruits (oranges, guavas), soft foods. Avoid dairy if it worsens congestion.'
    },
    {
        'Disease': 'Flu',
        'Description': 'A contagious respiratory illness caused by influenza viruses, characterized by fever, body aches, headache, and fatigue. Can lead to serious complications.',
        'Precautions': 'Get vaccinated annually, avoid close contact with sick people, cover coughs/sneezes, practice good hand hygiene, avoid touching face.',
        'Medications': 'Antivirals (e.g., oseltamivir, zanamivir) if prescribed early, pain relievers, cough and cold medicines, fever reducers.',
        'Diets': 'Hydrating foods (soups, broths, electrolyte drinks), soft and easily digestible foods, fresh fruits and vegetables. Avoid sugary drinks and heavy meals.'
    },
    {
        'Disease': 'Strep Throat',
        'Description': 'A bacterial infection (Streptococcus pyogenes) that can make your throat sore and scratchy, often with fever, red spots on the roof of the mouth, and swollen tonsils.',
        'Precautions': 'Avoid sharing utensils and food, wash hands thoroughly, complete the full course of antibiotics to prevent complications (e.g., rheumatic fever).',
        'Medications': 'Antibiotics (e.g., penicillin, amoxicillin), pain relievers for throat pain (e.g., ibuprofen, paracetamol), throat lozenges.',
        'Diets': 'Soft foods (yogurt, mashed potatoes, scrambled eggs), warm liquids (tea with honey), cold foods (popsicles, ice cream). Avoid spicy, acidic, or crunchy foods.'
    },
    {
        'Disease': 'Dengue',
        'Description': 'A mosquito-borne viral infection common in tropical and subtropical regions. Symptoms include high fever, severe headache, muscle and joint pains, rash, and nausea/vomiting.',
        'Precautions': 'Use mosquito repellents, wear protective clothing (long sleeves/pants), eliminate mosquito breeding sites (stagnant water in pots, tires), use bed nets.',
        'Medications': 'Pain relievers (paracetamol only), avoid aspirin and ibuprofen due to bleeding risk. Rest and fluid intake are crucial. No specific antiviral treatment.',
        'Diets': 'Stay hydrated with water, ORS (Oral Rehydration Solution), coconut water, and fruit juices. Eat light, easily digestible foods like porridges, soups, and boiled vegetables.'
    },
    {
        'Disease': 'Allergies',
        'Description': 'An immune system response to foreign substances (allergens) that are usually harmless. Symptoms vary by allergen but can include sneezing, runny nose, itchy eyes, skin rashes (hives), or wheezing.',
        'Precautions': 'Identify and avoid known allergens (pollen, dust mites, pet dander, certain foods). Keep windows closed during high pollen seasons, use air purifiers.',
        'Medications': 'Antihistamines (oral or nasal sprays), decongestants, corticosteroid nasal sprays, eye drops, epinephrine auto-injector for severe reactions (anaphylaxis).',
        'Diets': 'Avoid allergen foods if applicable. Some find anti-inflammatory foods (e.g., turmeric, ginger, omega-3 fatty acids) beneficial, but this is not a substitute for medical treatment.'
    },
    {
        'Disease': 'Gastroenteritis',
        'Description': 'Commonly known as "stomach flu," it is an inflammation of the stomach and intestines, usually caused by viruses, leading to nausea, vomiting, diarrhea, and abdominal cramps.',
        'Precautions': 'Frequent hand washing, especially after using the bathroom and before eating. Avoid contaminated food/water. Stay hydrated to prevent dehydration.',
        'Medications': 'Oral rehydration solutions (ORS) to prevent dehydration. Anti-diarrheal medications (e.g., loperamide) or anti-emetics might be used cautiously, but often not necessary for mild cases.',
        'Diets': 'BRAT diet (Bananas, Rice, Applesauce, Toast) initially. Gradually reintroduce bland, easily digestible foods. Avoid dairy, fatty, spicy, or sugary foods until symptoms subside. Plenty of fluids.'
    },
    {
        'Disease': 'Urinary Tract Infection',
        'Description': 'An infection in any part of the urinary system—kidneys, ureters, bladder, and urethra. Most infections involve the lower urinary tract (bladder and urethra). Symptoms include painful and frequent urination.',
        'Precautions': 'Drink plenty of water, cranberry juice (though scientific evidence is mixed), wipe from front to back, urinate after intercourse, avoid irritating feminine products.',
        'Medications': 'Antibiotics prescribed by a doctor (e.g., nitrofurantoin, trimethoprim-sulfamethoxazole). Pain relievers can help with discomfort.',
        'Diets': 'Drink plenty of water. Avoid bladder irritants like caffeine, alcohol, spicy foods, and acidic fruits/juices (except cranberry). Some find probiotics helpful.'
    },
    {
        'Disease': 'Migraine',
        'Description': 'A type of headache characterized by recurrent, moderate to severe headaches, often throbbing, usually on one side of the head, and often accompanied by nausea, vomiting, and sensitivity to light and sound.',
        'Precautions': 'Identify and avoid triggers (stress, certain foods, lack of sleep, hormonal changes). Maintain a regular sleep schedule, stay hydrated, manage stress.',
        'Medications': 'Pain relievers (NSAIDs, triptans), anti-nausea medications. Preventive medications (e.g., beta-blockers, antidepressants, CGRP inhibitors) may be prescribed for frequent migraines.',
        'Diets': 'Identify and avoid food triggers (e.g., aged cheese, red wine, processed meats, caffeine, artificial sweeteners). Some benefit from magnesium, riboflavin, or coenzyme Q10 supplements.'
    },
    {
        'Disease': 'Hypertension',
        'Description': 'Also known as high blood pressure, it is a common condition in which the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems, such as heart disease.',
        'Precautions': 'Regular blood pressure monitoring, healthy diet, regular exercise, limiting sodium intake, managing stress, avoiding smoking and excessive alcohol.',
        'Medications': 'Diuretics, ACE inhibitors, ARBs, calcium channel blockers, beta-blockers. Medications are tailored to the individual and often lifelong.',
        'Diets': 'DASH diet (Dietary Approaches to Stop Hypertension) focusing on fruits, vegetables, whole grains, lean protein, and low-fat dairy. Limit sodium, saturated/trans fats, and added sugars.'
    },
    {
        'Disease': 'Hypothyroidism',
        'Description': 'A condition in which your thyroid gland doesn\'t produce enough thyroid hormones. It can cause fatigue, weight gain, cold intolerance, dry skin, and depression.',
        'Precautions': 'Regular check-ups, especially if you have risk factors. Follow prescribed medication regimen strictly. Monitor symptoms and blood levels.',
        'Medications': 'Synthetic thyroid hormone levothyroxine is the primary treatment, taken daily. Dosage is adjusted based on blood tests.',
        'Diets': 'Balanced diet. Avoid excessive intake of goitrogenic foods (broccoli, cauliflower, cabbage) which can interfere with thyroid function if consumed raw and in very large amounts. Ensure adequate iodine (if deficient) and selenium intake.'
    },
    {
        'Disease': 'Anemia',
        'Description': 'A condition in which you lack enough healthy red blood cells to carry adequate oxygen to your body\'s tissues. Often caused by iron deficiency, leading to fatigue, weakness, and pale skin.',
        'Precautions': 'Ensure adequate intake of iron and vitamin C (enhances iron absorption). Address underlying causes of blood loss if present.',
        'Medications': 'Iron supplements (oral or IV), vitamin B12 injections (for pernicious anemia), folic acid supplements depending on the type of anemia.',
        'Diets': 'Iron-rich foods (red meat, poultry, fish, beans, lentils, spinach, fortified cereals). Vitamin C rich foods (citrus fruits, bell peppers) to aid iron absorption. Avoid tea/coffee with iron-rich meals.'
    },
    {
        'Disease': 'Anxiety Disorder',
        'Description': 'A group of mental disorders characterized by significant feelings of anxiety and fear. It can cause physical symptoms like heart palpitations, shortness of breath, trembling, and dizziness.',
        'Precautions': 'Stress management techniques (mindfulness, meditation, yoga), regular exercise, adequate sleep, limiting caffeine and alcohol, avoiding recreational drugs.',
        'Medications': 'Antidepressants (SSRIs, SNRIs), anti-anxiety medications (benzodiazepines, though usually for short-term use), beta-blockers. Often combined with therapy.',
        'Diets': 'Balanced diet. Some find reduction in caffeine and sugar helpful. Omega-3 fatty acids, probiotics, and foods rich in magnesium and B vitamins may support mood, but are not a primary treatment.'
    },
    {
        'Disease': 'Jaundice',
        'Description': 'A yellowish discoloration of the skin, mucous membranes, and whites of the eyes caused by increased levels of bilirubin in the blood. It often indicates an underlying liver, gallbladder, or blood disorder.',
        'Precautions': 'Jaundice itself is a symptom; precautions depend on the underlying cause. Avoid alcohol, liver-toxic medications, and exposure to hepatitis viruses. Seek prompt medical evaluation.',
        'Medications': 'Treatment depends on the underlying cause. This could include antivirals for hepatitis, antibiotics for infections, corticosteroids, or procedures to remove gallstones.',
        'Diets': 'Easy-to-digest foods, lean proteins, fruits, vegetables, and whole grains. Limit fatty, spicy, and processed foods. Avoid alcohol and hepatotoxic substances. Drink plenty of fluids.'
    },
    {
        'Disease': 'Dehydration',
        'Description': 'Occurs when you use or lose more fluid than you take in, and your body doesn\'t have enough water and other fluids to carry out its normal functions. Can be mild, moderate, or severe.',
        'Precautions': 'Drink plenty of fluids throughout the day, especially during exercise, hot weather, or illness. Replenish electrolytes after significant fluid loss.',
        'Medications': 'Oral rehydration solutions (ORS) for mild to moderate dehydration. IV fluids for severe cases under medical supervision.',
        'Diets': 'Water, broths, diluted fruit juices, fruits and vegetables with high water content (watermelon, cucumber). Avoid sugary sodas and excessive caffeine.'
    },
    {
        'Disease': 'Diabetes (Type 2)',
        'Description': 'A chronic condition that affects the way your body processes blood sugar (glucose). Your body either doesn\'t produce enough insulin, or it resists the effects of insulin.',
        'Precautions': 'Healthy diet, regular exercise, weight management, regular blood sugar monitoring, avoiding smoking.',
        'Medications': 'Metformin is often first-line. Other medications include sulfonylureas, GLP-1 receptor agonists, SGLT2 inhibitors. Insulin may be prescribed if needed.',
        'Diets': 'Balanced diet focusing on whole grains, lean proteins, healthy fats, and non-starchy vegetables. Limit refined carbohydrates, sugary drinks, and processed foods. Count carbohydrates.'
    },
    {
        'Disease': 'Kidney Stones',
        'Description': 'Hard deposits of minerals and salts that form inside your kidneys. Can cause severe pain when they pass through the urinary tract. Can lead to painful urination and blood in urine.',
        'Precautions': 'Drink plenty of water throughout the day. Reduce intake of sodium, animal protein, and oxalate-rich foods (for calcium oxalate stones).',
        'Medications': 'Pain relievers (NSAIDs). Alpha-blockers to help relax ureter muscles. Medical procedures (lithotripsy, ureteroscopy) for larger stones.',
        'Diets': 'High fluid intake. For calcium oxalate stones: limit oxalate (spinach, almonds, rhubarb, chocolate), reduce sodium, adequate calcium. For uric acid stones: reduce animal protein.'
    },
    {
        'Disease': 'Pneumonia',
        'Description': 'An infection that inflames the air sacs in one or both lungs, which may fill with fluid or pus. Symptoms include cough with phlegm, fever, chills, and difficulty breathing.',
        'Precautions': 'Vaccinations (flu, pneumococcal), frequent hand washing, avoiding smoking, managing chronic health conditions.',
        'Medications': 'Antibiotics (for bacterial pneumonia), antivirals (for viral pneumonia), anti-fungals (for fungal pneumonia). Cough medicine, pain relievers, and fever reducers.',
        'Diets': 'Maintain good hydration. Eat nutrient-rich foods to support recovery. Warm soups and easily digestible meals.'
    },
    {
        'Disease': 'Bronchitis',
        'Description': 'Inflammation of the lining of your bronchial tubes, which carry air to and from your lungs. Often causes a cough with mucus, shortness of breath, and chest discomfort.',
        'Precautions': 'Avoid smoking and secondhand smoke. Get vaccinated against flu and pneumonia. Wash hands frequently.',
        'Medications': 'Cough suppressants, bronchodilators (for wheezing/shortness of breath), pain relievers/fever reducers. Antibiotics only if bacterial infection is confirmed.',
        'Diets': 'Stay well hydrated. Avoid foods that may trigger coughing or mucus production. Nutrient-dense foods to support the immune system.'
    },
    {
        'Disease': 'Asthma',
        'Description': 'A chronic respiratory condition causing inflammation and narrowing of the airways, leading to wheezing, shortness of breath, chest tightness, and coughing.',
        'Precautions': 'Identify and avoid triggers (allergens, irritants, exercise). Follow an asthma action plan. Get vaccinated for flu and pneumonia.',
        'Medications': 'Rescue inhalers (bronchodilators) for quick relief. Long-term control medications (inhaled corticosteroids, LABAs, biologics) to prevent attacks.',
        'Diets': 'No specific asthma diet, but a healthy, balanced diet is recommended. Some find relief by avoiding trigger foods or embracing anti-inflammatory diets. Avoid sulfites if sensitive.'
    },
    {
        'Disease': 'Eczema',
        'Description': 'A condition that causes your skin to become dry, itchy, and inflamed. It\'s a common chronic skin condition, also known as atopic dermatitis.',
        'Precautions': 'Moisturize regularly, avoid harsh soaps and irritants, take lukewarm baths, identify and avoid triggers (allergens, stress).',
        'Medications': 'Topical corticosteroids, calcineurin inhibitors, antihistamines for itching, phototherapy, sometimes oral corticosteroids or biologics for severe cases.',
        'Diets': 'Identify and avoid food triggers if present (e.g., dairy, eggs, peanuts, soy). Some find anti-inflammatory diets beneficial. Consider probiotics.'
    },
    {
        'Disease': 'Psoriasis',
        'Description': 'A chronic autoimmune condition that causes rapid buildup of skin cells, resulting in thick, silvery scales and itchy, dry, red patches that can be painful.',
        'Precautions': 'Moisturize, avoid triggers (stress, skin injury, infections, certain medications), warm baths with oils, sun exposure (in moderation).',
        'Medications': 'Topical corticosteroids, vitamin D analogues, retinoids. Systemic medications (methotrexate, cyclosporine, biologics) for severe cases. Phototherapy.',
        'Diets': 'Some find an anti-inflammatory diet (rich in fruits, vegetables, lean protein, omega-3s, low in red meat, dairy, refined carbs) can help. Avoiding alcohol is often recommended.'
    },
    {
        'Disease': 'Arthritis (Osteo)',
        'Description': 'The most common form of arthritis, characterized by the breakdown of cartilage that cushions the ends of bones. Leads to joint pain, stiffness, and loss of flexibility.',
        'Precautions': 'Maintain a healthy weight, regular low-impact exercise (swimming, cycling), use assistive devices, protect joints from injury.',
        'Medications': 'Pain relievers (paracetamol, NSAIDs), topical creams, corticosteroid injections, hyaluronic acid injections. Joint replacement surgery for severe cases.',
        'Diets': 'Focus on anti-inflammatory foods (omega-3 fatty acids, fruits, vegetables, whole grains). Limit processed foods, added sugars, and saturated/trans fats. Maintain a healthy weight.'
    },
    {
        'Disease': 'Depression',
        'Description': 'A common and serious medical illness that negatively affects how you feel, the way you think, and how you act. It causes feelings of sadness and/or a loss of interest in activities you once enjoyed.',
        'Precautions': 'Seek professional help, maintain social connections, engage in regular exercise, ensure adequate sleep, practice stress reduction techniques, avoid alcohol/drugs.',
        'Medications': 'Antidepressants (SSRIs, SNRIs, tricyclics, MAOIs). Medication is often used in combination with psychotherapy (CBT, interpersonal therapy).',
        'Diets': 'Balanced, nutrient-rich diet. Some research suggests a link between gut health and mood; probiotics and fermented foods might be beneficial. Limiting processed foods and sugar.'
    },
    {
        'Disease': 'Vertigo (BPPV)',
        'Description': 'Benign paroxysmal positional vertigo (BPPV) is one of the most common causes of vertigo — the sudden sensation that you\'re spinning or that the inside of your head is spinning. It\'s triggered by specific head movements.',
        'Precautions': 'Careful head movements, especially when getting out of bed or looking up. Avoid positions that trigger vertigo. Perform specific exercises (e.g., Epley maneuver) under guidance.',
        'Medications': 'Usually not treated with medication for BPPV itself. Anti-nausea medications or vestibular suppressants may be used for symptom relief in short-term severe cases.',
        'Diets': 'No specific diet for BPPV, but a balanced diet generally supports overall health. Limiting caffeine and alcohol might help if they exacerbate dizziness for some individuals.'
    },
    {
        'Disease': 'Food Poisoning',
        'Description': 'Illness caused by food contaminated with bacteria, viruses, parasites, or toxins. Symptoms include nausea, vomiting, diarrhea, abdominal cramps, and sometimes fever.',
        'Precautions': 'Practice safe food handling: wash hands, cook foods to proper temperatures, avoid cross-contamination, refrigerate promptly. Avoid unpasteurized products.',
        'Medications': 'Oral rehydration solutions (ORS) to prevent dehydration. Anti-diarrheal medications might be used cautiously. Antibiotics only if bacterial infection is confirmed and severe.',
        'Diets': 'Focus on clear liquids and bland foods (BRAT diet) as tolerated. Gradually reintroduce solid foods. Avoid dairy, fatty, spicy, and highly fibrous foods until symptoms subside.'
    },
]
disease_info = pd.DataFrame(disease_info_raw_data)


# --- Hardcoded Medication Interactions (for demo) ---
MEDICATION_INTERACTIONS = {
    "ibuprofen": {
        "conditions_interact": ["Hypertension", "Kidney Stones", "Asthma"],
        "meds_interact": ["aspirin", "warfarin", "furosemide"],
        "warning": "Ibuprofen can increase blood pressure, worsen kidney function, or trigger asthma attacks in sensitive individuals. It can also interact with blood thinners like warfarin and diuretics like furosemide, increasing bleeding risk or reducing diuretic effect. Consult your doctor."
    },
    "paracetamol": {
        "conditions_interact": ["Liver Disease"],
        "meds_interact": ["warfarin"],
        "warning": "Excessive paracetamol (acetaminophen) can cause liver damage, especially if you have existing liver conditions or consume alcohol. It can also increase the effect of blood thinners like warfarin. Do not exceed recommended dosage."
    },
    "dextromethorphan": { # Common cough suppressant
        "conditions_interact": ["Depression", "Anxiety Disorder"], # Due to potential interaction with MAOIs/SSRIs if user is on them
        "meds_interact": ["SSRIs", "MAOIs", "opioids"], # Generic terms for demo
        "warning": "Dextromethorphan can interact with certain antidepressants (SSRIs, MAOIs) and other medications, potentially leading to serotonin syndrome. Avoid if you are on psychiatric medications, unless advised by a doctor."
    },
    "decongestants": { # E.g., pseudoephedrine, phenylephrine
        "conditions_interact": ["Hypertension", "Heart Palpitations", "Anxiety Disorder"],
        "meds_interact": ["MAOIs", "beta-blockers"],
        "warning": "Decongestants can raise blood pressure and heart rate. Use with caution if you have hypertension, heart conditions, or anxiety. Avoid if on certain antidepressants (MAOIs) or beta-blockers without medical advice."
    },
    "antibiotics": { # General class
        "conditions_interact": ["Kidney Stones", "Liver Disease"], # Some antibiotics are contraindicated/need dose adjustment
        "meds_interact": ["warfarin", "oral contraceptives"], # Broad examples
        "warning": "Antibiotics can interact with various medications and may require dose adjustments for certain conditions. They can reduce the effectiveness of oral contraceptives and increase the effect of warfarin. Always inform your doctor about all medications you are taking."
    },
    "antihistamines": { # Older, sedating ones
        "conditions_interact": ["Glaucoma", "Prostate Enlargement"],
        "meds_interact": ["sedatives", "alcohol"],
        "warning": "Some older antihistamines can cause drowsiness and interact with other sedating medications or alcohol. Use with caution if you have glaucoma or prostate enlargement."
    },
    "triptans": { # For Migraine
        "conditions_interact": ["Hypertension", "Heart Disease"],
        "meds_interact": ["SSRIs", "MAOIs"],
        "warning": "Triptans can affect blood pressure and are generally avoided in those with uncontrolled hypertension or heart disease. They can also interact with SSRIs/MAOIs, increasing risk of serotonin syndrome. Discuss with your doctor."
    }
}

# --- Mapping Diseases to Medical Specialties ---
MEDICAL_SPECIALTIES = {
    'Common Cold': 'General Physician',
    'Flu': 'General Physician',
    'Strep Throat': 'General Physician / ENT Specialist',
    'Dengue': 'General Physician / Infectious Disease Specialist',
    'Allergies': 'Allergist / Immunologist',
    'Gastroenteritis': 'General Physician / Gastroenterologist',
    'Urinary Tract Infection': 'General Physician / Urologist',
    'Migraine': 'Neurologist',
    'Hypertension': 'General Physician / Cardiologist',
    'Hypothyroidism': 'Endocrinologist',
    'Anemia': 'General Physician / Hematologist',
    'Anxiety Disorder': 'Psychiatrist / Psychologist',
    'Jaundice': 'Gastroenterologist / Hepatologist',
    'Dehydration': 'General Physician',
    'Diabetes (Type 2)': 'Endocrinologist',
    'Kidney Stones': 'Urologist',
    'Pneumonia': 'Pulmonologist / General Physician',
    'Bronchitis': 'Pulmonologist / General Physician',
    'Asthma': 'Pulmonologist / Allergist',
    'Eczema': 'Dermatologist',
    'Psoriasis': 'Dermatologist',
    'Arthritis (Osteo)': 'Orthopedist / Rheumatologist',
    'Depression': 'Psychiatrist / Psychologist',
    'Vertigo (BPPV)': 'ENT Specialist / Neurologist',
    'Food Poisoning': 'General Physician / Gastroenterologist',
    # Default for unknown diseases
    'Default': 'General Physician'
}


# --- Utility Function to Check for Interactions ---
def check_for_interactions(recommended_medications_str, current_user_profile):
    warnings = []
    # Convert recommended medications string into a searchable format (e.g., lowercase list of keywords)
    recommended_meds_keywords = [m.strip().lower() for m in recommended_medications_str.replace(',', ' ').split()]
    # Add common names that might appear in description but map to key
    if "ibuprofen" in recommended_meds_keywords or "nsaids" in recommended_meds_keywords:
        recommended_meds_keywords.append("ibuprofen")
    if "paracetamol" in recommended_meds_keywords or "acetaminophen" in recommended_meds_keywords:
        recommended_meds_keywords.append("paracetamol")
    if "dextromethorphan" in recommended_meds_keywords or "cough suppressant" in recommended_meds_keywords:
        recommended_meds_keywords.append("dextromethorphan")
    if "decongestants" in recommended_meds_keywords or "pseudoephedrine" in recommended_meds_keywords or "phenylephrine" in recommended_meds_keywords:
        recommended_meds_keywords.append("decongestants")
    if "antibiotics" in recommended_meds_keywords:
        recommended_meds_keywords.append("antibiotics")
    if "antihistamines" in recommended_meds_keywords:
        recommended_meds_keywords.append("antihistamines")
    if "triptans" in recommended_meds_keywords:
        recommended_meds_keywords.append("triptans")


    user_current_meds = [m.lower() for m in current_user_profile.get('current_medications', [])]
    user_existing_conditions = [c.lower() for c in current_user_profile.get('existing_conditions', [])]

    for med_name, interaction_data in MEDICATION_INTERACTIONS.items():
        # Check if the "recommended" med matches one of our interaction keys
        if med_name.lower() in recommended_meds_keywords:
            for condition in interaction_data.get("conditions_interact", []):
                if condition.lower() in user_existing_conditions:
                    warnings.append(f"**Potential Interaction (Disease):** {interaction_data['warning']}")
                    break
            for user_med in user_current_meds:
                for interacts_with_med in interaction_data.get("meds_interact", []):
                    if interacts_with_med.lower() == user_med:
                        warnings.append(f"**Potential Interaction (Medication):** {interaction_data['warning']}")
                        break
                if warnings: # If a warning was added, no need to check further for this recommended med
                    break
    return list(set(warnings)) # Return unique warnings

# --- Flask Routes ---

@app.route('/')
def login_or_register():
    """Initial page for user login/registration."""
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']

    if username in user_profiles:
        session['username'] = username
        flash(f'Welcome back, {username}!', 'success')
        return redirect(url_for('home'))
    else:
        user_profiles[username] = {
            'age': None,
            'gender': None,
            'existing_conditions': [],
            'allergies': [],
            'current_medications': []
        }
        user_history[username] = []
        session['username'] = username
        flash(f'Welcome, {username}! Please complete your profile.', 'info')
        return redirect(url_for('edit_profile'))

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('login_or_register'))

@app.route('/home')
def home():
    if 'username' not in session:
        flash('Please login to access this page.', 'warning')
        return redirect(url_for('login_or_register'))
    username = session['username']
    user_profile = user_profiles.get(username)

    if not user_profile:
        flash('User profile not found. Please login again.', 'danger')
        session.pop('username', None)
        return redirect(url_for('login_or_register'))

    return render_template('index.html', all_symptoms=all_symptoms, user_profile=user_profile,
                           customer_care_phone=CUSTOMER_CARE_PHONE, customer_care_email=CUSTOMER_CARE_EMAIL)

@app.route('/profile', methods=['GET', 'POST'])
def edit_profile():
    if 'username' not in session:
        flash('Please login to access your profile.', 'warning')
        return redirect(url_for('login_or_register'))

    username = session['username']
    current_user_profile = user_profiles.get(username)
    if not current_user_profile:
        flash('User profile not found. Please login again.', 'danger')
        session.pop('username', None)
        return redirect(url_for('login_or_register'))

    if request.method == 'POST':
        current_user_profile['age'] = request.form.get('age', type=int)
        current_user_profile['gender'] = request.form.get('gender')
        current_user_profile['existing_conditions'] = [
            cond.strip() for cond in request.form.get('existing_conditions', '').split(',') if cond.strip()
        ]
        current_user_profile['allergies'] = [
            allergy.strip() for allergy in request.form.get('allergies', '').split(',') if allergy.strip()
        ]
        current_user_profile['current_medications'] = [
            med.strip() for med in request.form.get('current_medications', '').split(',') if med.strip()
        ]

        flash('Profile updated successfully!', 'success')
        return jsonify({"message": "Profile updated successfully!"}), 200

    return render_template('profile.html', user_profile=current_user_profile)

@app.route('/recommend', methods=['POST'])
def recommend():
    if 'username' not in session:
        return jsonify({"error": "User not logged in."}), 401

    username = session['username']
    current_user_profile = user_profiles.get(username)
    if not current_user_profile:
        return jsonify({"error": "User profile not found."}), 401

    selected_symptoms = request.json.get('symptoms', [])
    print(f"Received symptoms for {username}: {selected_symptoms}")

    user_symptoms_vector = [0] * len(all_symptoms)
    for symptom in selected_symptoms:
        if symptom in all_symptoms:
            user_symptoms_vector[all_symptoms.index(symptom)] = 1
        else:
            print(f"Warning: Symptom '{symptom}' not recognized by model and will be ignored.")

    user_input_df = pd.DataFrame([user_symptoms_vector], columns=all_symptoms)

    try:
        predicted_disease = model.predict(user_input_df)[0]
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": "Prediction failed. Please try again."}), 500

    disease_details = disease_info[disease_info['Disease'] == predicted_disease]

    result = {}
    recommended_medications_text = ""
    if not disease_details.empty:
        result = {
            "predicted_disease": predicted_disease,
            "description": disease_details['Description'].iloc[0],
            "precautions": disease_details['Precautions'].iloc[0],
            "medications": disease_details['Medications'].iloc[0],
            "diets": disease_details['Diets'].iloc[0]
        }
        recommended_medications_text = disease_details['Medications'].iloc[0]
    else:
        result = {
            "predicted_disease": predicted_disease,
            "description": "Information not available for this disease.",
            "precautions": "Consult a doctor for further advice.",
            "medications": "Consult a doctor for suitable medications.",
            "diets": "Consult a doctor for dietary recommendations."
        }
        recommended_medications_text = "N/A"

    # --- Determine Recommended Specialty ---
    recommended_specialty = MEDICAL_SPECIALTIES.get(predicted_disease, MEDICAL_SPECIALTIES['Default'])
    result['recommended_specialty'] = recommended_specialty

    # --- Personalization & Professional Guidance Logic ---
    personal_guidance_messages = []
    user_existing_conditions = [c.lower() for c in current_user_profile.get('existing_conditions', [])]
    user_allergies = [a.lower() for a in current_user_profile.get('allergies', [])]
    user_current_meds = [m.lower() for m in current_user_profile.get('current_medications', [])]
    user_age = current_user_profile.get('age')

    personal_guidance_messages.append("<h3>Professional Health Guidance:</h3>")
    personal_guidance_messages.append("<p><strong>Important: This system offers general recommendations. Always consult a qualified healthcare professional for a precise diagnosis and personalized treatment plan.</strong></p>")

    # Add general advice based on age
    if user_age is not None:
        if user_age >= 65:
            personal_guidance_messages.append("<p><strong>For Older Adults:</strong> Your body may react differently to illnesses and medications. It's especially important to seek medical advice for new or worsening symptoms. Discuss all medications, including over-the-counter ones, with your doctor.</p>")
        elif user_age < 18:
            personal_guidance_messages.append("<p><strong>For Children/Teenagers:</strong> Always consult a pediatrician for accurate diagnosis and appropriate treatment. Self-medication can be dangerous for younger individuals.</p>")
    else:
        personal_guidance_messages.append("<p><strong>Profile Incomplete:</strong> Please update your age in your profile for more tailored guidance.</p>")


    # Disease-specific guidance based on profile
    if predicted_disease == 'Flu':
        if 'diabetes (type 2)' in user_existing_conditions:
            personal_guidance_messages.append("<p><strong>Flu & Diabetes:</strong> If you have diabetes, the flu can significantly affect your blood sugar levels. Monitor your blood sugar closely and contact your doctor for specific guidance on managing your diabetes during this illness.</p>")
        if user_age is not None and user_age >= 65:
            personal_guidance_messages.append("<p><strong>Flu in Older Adults:</strong> Flu can lead to severe complications like pneumonia in older adults. Consider getting vaccinated annually and seek medical attention early if symptoms are severe.</p>")
        if 'asthma' in user_existing_conditions:
             personal_guidance_messages.append("<p><strong>Flu & Asthma:</strong> The flu can trigger asthma attacks. Keep your asthma action plan ready and consult your doctor for flu management.</p>")
        if 'fever' in selected_symptoms and selected_symptoms.count('Fever') == 1 and 'high fever' in selected_symptoms: # Example of looking for specific phrasing or multiple symptoms
            personal_guidance_messages.append("<p><strong>High Fever Alert:</strong> A high fever (e.g., above 102°F or 39°C) especially with flu-like symptoms warrants medical consultation, particularly if it persists or is accompanied by severe body aches or difficulty breathing.</p>")

    if predicted_disease == 'Strep Throat':
        if 'fever' in selected_symptoms and 'rash' in selected_symptoms:
            personal_guidance_messages.append("<p><strong>Strep Throat with Rash:</strong> A rash with strep throat could indicate Scarlet Fever, which requires prompt medical attention and full antibiotic course to prevent complications.</p>")
        personal_guidance_messages.append("<p><strong>Antibiotic Importance:</strong> If diagnosed with Strep Throat, it is critical to complete the full course of antibiotics as prescribed by a doctor to prevent serious complications like rheumatic fever.</p>")

    if predicted_disease == 'Dengue':
        if any(s in ['abdominal pain', 'vomiting', 'bleeding', 'bruising easily'] for s in selected_symptoms):
            personal_guidance_messages.append("<p><strong>Dengue Warning Signs:</strong> If you experience severe abdominal pain, persistent vomiting, bleeding from gums/nose, rapid breathing, or unusual fatigue with suspected Dengue, **seek emergency medical care immediately** as these are warning signs of severe dengue.</p>")
        personal_guidance_messages.append("<p><strong>Hydration for Dengue:</strong> Maintaining adequate hydration is paramount in Dengue fever. Drink plenty of fluids like water, ORS, and fresh juices. Avoid dark-colored drinks that might mask internal bleeding.</p>")


    if predicted_disease == 'Hypertension':
        if 'headache' in selected_symptoms and 'vision blurry' in selected_symptoms:
            personal_guidance_messages.append("<p><strong>Hypertension Crisis Warning:</strong> Severe headache and blurred vision in someone with hypertension can be signs of a hypertensive crisis. **Seek immediate medical attention.** Do not self-medicate.</p>")
        personal_guidance_messages.append("<p><strong>Blood Pressure Monitoring:</strong> If you are diagnosed with hypertension, regular monitoring of your blood pressure is essential. Follow your doctor's recommendations for medication and lifestyle changes diligently.</p>")

    if predicted_disease == 'Urinary Tract Infection':
        if 'blood in urine' in selected_symptoms or 'fever' in selected_symptoms or 'back pain' in selected_symptoms:
            personal_guidance_messages.append("<p><strong>Possible Kidney Infection:</strong> If your UTI symptoms include fever, chills, back pain, or blood in urine, it may indicate a kidney infection, which requires urgent medical care.</p>")

    if predicted_disease == 'Gastroenteritis':
        if 'diarrhea' in selected_symptoms and selected_symptoms.count('Diarrhea') > 3 and user_age is not None and user_age < 5:
             personal_guidance_messages.append("<p><strong>Dehydration in Children:</strong> For young children with severe or persistent diarrhea/vomiting, dehydration can occur rapidly. Seek medical attention immediately.</p>")
        if 'vomiting' in selected_symptoms and selected_symptoms.count('Vomiting') > 5:
            personal_guidance_messages.append("<p><strong>Severe Vomiting:</strong> Persistent or severe vomiting can lead to dehydration. Focus on small, frequent sips of ORS. If vomiting prevents fluid intake, seek medical help.</p>")

    if predicted_disease == 'Anemia':
        personal_guidance_messages.append("<p><strong>Anemia Investigation:</strong> Anemia can be a symptom of an underlying condition. It's crucial to consult a doctor to determine the cause of your anemia and get appropriate treatment, not just take iron supplements blindly.</p>")

    if predicted_disease == 'Depression' or predicted_disease == 'Anxiety Disorder':
        personal_guidance_messages.append("<p><strong>Mental Health Support:</strong> For symptoms of depression or anxiety, professional help from a therapist or psychiatrist is highly recommended. These conditions often require tailored therapy and/or medication for effective management.</p>")
        personal_guidance_messages.append("<p><strong>Crisis Support:</strong> If you are experiencing thoughts of self-harm or severe distress, please seek immediate help. You can contact a local crisis hotline or emergency services.</p>")

    # Allergy warning (already present, but reinforcing here for context)
    for allergy in user_allergies:
        if allergy.lower() in recommended_medications_text.lower():
            personal_guidance_messages.append(f"<h3>Personal Health Advisory:</h3><p><strong>Severe Allergy Warning:</strong> You have listed an allergy to '{allergy}'. The recommended medications might contain substances you are allergic to. **DO NOT take any medication before consulting your doctor or pharmacist and confirming it is safe for you.** An allergic reaction can be life-threatening.</p>")
            break # Only need one general allergy warning for meds

    # Generic advice if profile is incomplete
    if not all([current_user_profile.get('age'), current_user_profile.get('gender'), current_user_profile.get('existing_conditions'), current_user_profile.get('allergies'), current_user_profile.get('current_medications')]):
        personal_guidance_messages.append("<p><strong>Incomplete Profile:</strong> Please complete your profile (age, gender, existing conditions, allergies, current medications) for the most accurate and personalized professional guidance. Missing information can lead to less relevant advice.</p>")

    # Add the recommended specialty to guidance
    personal_guidance_messages.append(f"<p><strong>Recommended Specialist:</strong> Based on your symptoms, it is advisable to consult a **{recommended_specialty}** for proper diagnosis and treatment.</p>")

    # Add customer care information to guidance (only in index page result)
    personal_guidance_messages.append(f"<p>For further assistance or to discuss your results, you can contact our Customer Care service at <strong><a href='tel:{CUSTOMER_CARE_PHONE}'>{CUSTOMER_CARE_PHONE}</a></strong> or email us at <a href='mailto:{CUSTOMER_CARE_EMAIL}'>{CUSTOMER_CARE_EMAIL}</a>.</p>")


    # FINAL IMPORTANT DISCLAIMER
    personal_guidance_messages.append("<p><strong>ALWAYS CONSULT A DOCTOR:</strong> This automated system provides preliminary information. It is crucial to seek advice from a qualified medical professional for any health concerns, diagnosis, or before starting or changing any treatment.</p>")

    result['personal_guidance'] = "".join(personal_guidance_messages)

    # --- Medication Interaction Checker ---
    interaction_warnings = check_for_interactions(recommended_medications_text, current_user_profile)
    result['interaction_warnings'] = "".join([f"<p>{w}</p>" for w in interaction_warnings]) if interaction_warnings else ""

    # --- Store in user history (in-memory) ---
    user_history.setdefault(username, []).append({
        'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
        'symptoms': selected_symptoms,
        'predicted_disease': predicted_disease,
        'recommended_specialty': recommended_specialty, # Store specialty in history
        'personal_guidance_provided': bool(personal_guidance_messages),
        'interaction_warnings': interaction_warnings
    })

    return jsonify(result)

@app.route('/history')
def view_history():
    if 'username' not in session:
        flash('Please login to view your history.', 'warning')
        return redirect(url_for('login_or_register'))
    username = session['username']
    history_for_user = user_history.get(username, [])

    sorted_history = sorted(history_for_user, key=lambda x: x['timestamp'], reverse=True)

    return render_template('history.html', history=sorted_history)

if __name__ == '__main__':
    print("Starting Flask server.")
    app.run(debug=True) 