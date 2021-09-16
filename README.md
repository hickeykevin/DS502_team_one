# DS502_team_one
Repository for DS501 Final Group Project


## List of Variables in Data for Analysis:

#### [Outcome Variable] 11 MDD Related Clinical Measures: 
Measures are continuous in raw dataset. Measures are binary in cleaned dataset based on clinician chosen cutoff (with 1 being high).
- sBCog45S: Brief Risk resilience Index for Screening Emotional/Cognitive Resilience
- sBNeg45S: Brief Risk resilience Index for Screening Negativity Bias
- sBSoc45S: Brief Risk resilience Index for Screening Social Connectedness
- sERQreap: Emotion Regulation Cognitive Reappraisal
- sERQsupp: Emotion Regulation Expressive Suppression
- sAnx42: Depression Anxiety Stress Scale Anxiety subscale
- sDepr42: Depression Anxiety Stress Scale Depression subscale
- sStres42: Depression Anxiety Stress Scale Stress subscale
- HDRS17: Hamilton Depression Rating Scale (17-item) Total
- sQIDS_tot: Quick Inventory Depressive Symptoms Total
- SOFAS_RAT: Social and Occupational Functinoal Assessment (SOFAS)

#### [Outcome Variable] 12 Cluster Derived Phenotypes (composition of questionnaire items from MDD surveys):
Measures are continuous in raw dataset scaled from 0-1 (with 1 being more MDD like). Measures are binary in cleaned dataset based on median (with 1 being high).
- Agitation_Slowing
- Disorganized_Pessimistic
- EmotionReappraisal
- Hopeless_Anhedonia_Avoition
- Insomnia_Energy
- Irritable_Agitated
- NegEmoImage
- Nervous_Anticipatory
- Sad_DepressedMood
- Social_Emotiol
- Social_Productive
- Somatic_PosSymp_Panic

#### [Predictor Variable] 10 Demographics
- sex: categorical M and F
- Age: age in years numeric type with 2 decimal places
- Years_Education
- sBMI: BMI score
- MDD_DUR: Duration on depression in years
- ANXIOUS_TYPE: Anxious depression subtype
- ATYPICAL_TYPE: Atypical depression subtype
- MELANCHOLIC_TYPE: Melancholic depression subtype
- Treatment_group: Drug treatment categories of 1, 2, 3 for MDD and treatment category of 4 for Control
- sELSTOT: Early Life Stress Total Score

#### [Predictor Variable] 10 Event-related potential waveform (ERP)
Averaging brain activity in EEG as a numeric variable. 
Two regions: P300 (part of brain for decision making and auditory oddball stimuli) and N1 (Visual evoked potential. attention). 
Measures for std and trg for min/max.
- std_N1_amp_min_pub_Fz
- std_N1_amp_min_pub_Cz 
- trg_N1_amp_min_pub_Fz 
- trg_N1_amp_min_pub_Cz 
- std_P300_amp_max_Fz 
- std_P300_amp_max_Cz 
- std_P300_amp_max_Pz 
- trg_P300_amp_max_Cz 
- trg_P300_amp_max_Fz 
- trg_P300_amp_max_Pz

#### [Predictor Variable] Genetic SNPs. 
118 SNPs in raw data. 101 SNPs in cleaned data after removing low cardinality SNPs. May need to remove even more SNPs in cleaned data.
Number of alleles as a numeric variable from 0, 1, 2.
Variable names start with rs. Last character is the DNA base pair for the SNP






