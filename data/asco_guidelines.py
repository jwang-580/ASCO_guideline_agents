"""
ASCO Guidelines Data
This module contains the URLs and summaries of ASCO guidelines.
"""

# Dictionary mapping guideline names to their publication year and URL
guideline_urls = {
   "melanoma_cancer_1": ['2023', 'https://ascopubs.org/doi/pdfdirect/10.1200/JCO.23.01136'],
    "breast_cancer_1": ['2024', 'https://ascopubs.org/doi/pdfdirect/10.1200/JCO.23.02225'],
    "breast_cancer_2": ['2023', 'https://ascopubs.org/doi/pdfdirect/10.1200/JCO.22.02864'],
    "breast_cancer_3": ['2023', 'https://ascopubs.org/doi/pdfdirect/10.1200/JCO.23.00638'],
    "breast_cancer_4": ['2023', 'https://ascopubs.org/doi/pdfdirect/10.1200/JCO.22.02807'],
    "breast_cancer_5": ['2023', 'https://ascopubs.org/doi/pdfdirect/10.1200/JCO.22.01533'],
    "breast_cancer_6": ['2022', 'https://ascopubs.org/doi/pdfdirect/10.1200/JCO.22.01063'],
    "breast_cancer_7": ['2022', 'https://ascopubs.org/doi/pdfdirect/10.1200/JCO.22.00069'],
    "breast_cancer_8": ['2022', 'https://ascopubs.org/doi/pdfdirect/10.1200/JCO.21.02647'],
    "breast_cancer_9": ['2022', 'https://ascopubs.org/doi/pdfdirect/10.1200/JCO.22.00503'],
    "breast_cancer_10": ['2021', 'https://ascopubs.org/doi/pdfdirect/10.1200/JCO.21.02677'],
    "breast_cancer_11": ['2021', 'https://ascopubs.org/doi/pdfdirect/10.1200/JCO.21.01532'],
    "breast_cancer_12": ['2021', 'https://ascopubs.org/doi/pdfdirect/10.1200/JCO.21.01374'],
    "breast_cancer_13": ['2021', 'https://ascopubs.org/doi/pdfdirect/10.1200/JCO.21.01392'],
    "breast_cancer_14": ['2021', 'https://ascopubs.org/doi/pdfdirect/10.1200/JCO.21.00934'],
    "breast_cancer_15": ['2021', 'https://ascopubs.org/doi/pdfdirect/10.1200/JCO.20.03399'],
    "gi_cancer_1": ['2024', 'https://ascopubs.org/doi/pdfdirect/10.1200/JCO-24-02120'],
    "gi_cancer_2": ['2023', 'https://ascopubs.org/doi/pdfdirect/10.1200/JCO.22.02331'],
    "gi_cancer_3": ['2022', 'https://ascopubs.org/doi/pdfdirect/10.1200/JCO.22.01690'],
    "gi_cancer_4": ['2021', 'https://ascopubs.org/doi/pdfdirect/10.1200/JCO.21.02538'],
    "gu_cancer_1": ['2023', 'https://ascopubs.org/doi/pdfdirect/10.1200/JCO.23.00155'],
    "gu_cancer_2": ['2022', 'https://ascopubs.org/doi/pdfdirect/10.1200/JCO.22.00868'],
    "gynecologic_cancer_1": ['2022', 'https://ascopubs.org/doi/pdfdirect/10.1200/JCO.22.01934'],
    "headneck_cancer_1": ['2024', 'https://ascopubs.org/doi/pdfdirect/10.1200/JCO.23.02750'],
    "headneck_cancer_2": ['2022', 'https://ascopubs.org/doi/pdfdirect/10.1200/JCO.22.02328'],
    "headneck_cancer_3": ['2021', 'https://ascopubs.org/doi/pdfdirect/10.1200/JCO.21.00449'],
    "headneck_cancer_4": ['2021', 'https://ascopubs.org/doi/pdfdirect/10.1200/JCO.20.03237'],
    "neurooncology_cancer_1": ['2022', 'https://ascopubs.org/doi/pdfdirect/10.1200/JCO.22.00333'],
    "neurooncology_cancer_2": ['2021', 'https://ascopubs.org/doi/pdfdirect/10.1200/JCO.21.02314'],
    "neurooncology_cancer_3": ['2021', 'https://ascopubs.org/doi/pdfdirect/10.1200/JCO.21.02036'],
    "lung_cancer_1": ['2024', 'https://ascopubs.org/doi/pdfdirect/10.1200/JCO-24-02133'],
    "lung_cancer_4": ['2023', 'https://ascopubs.org/doi/pdfdirect/10.1200/JCO.23.02746'],
    "lung_cancer_5": ['2022', 'https://ascopubs.org/doi/pdfdirect/10.1200/JCO.22.00051'],
    "lung_cancer_6": ['2021', 'https://ascopubs.org/doi/pdfdirect/10.1200/JCO.21.02528'],
}

# Dictionary containing summaries of each guideline
guideline_summaries = {
    "melanoma_cancer_1": "Updates for melanoma systemic therapy include neoadjuvant pembrolizumab for resectable stage IIIB-IV disease and adjuvant therapy options for stage IIB-IV cutaneous melanoma. Nivolumab with ipilimumab is preferred for metastatic cases, while talimogene laherparepvec is no longer recommended for BRAF wild-type melanoma post anti–PD-1 progression. Uveal melanoma recommendations are also incorporated.",
    "breast_cancer_1": "The guideline emphasizes germline BRCA1/2 testing for all newly diagnosed breast cancer patients under 65 and selectively over 65 based on history or treatment needs. It includes testing for other high-penetrance genes for patients with family history and recommends post-test counseling for pathogenic variants. This update consolidates earlier guidance and adapts to new genetic testing technologies.",
    "breast_cancer_2": "HER2 testing guidelines reaffirm prior recommendations while addressing the emerging relevance of HER2-low status for therapies like trastuzumab deruxtecan. It clarifies best practices for distinguishing IHC 0 and 1+ results in testing but does not introduce new HER2 categories. This update acknowledges expanded indications without revising fundamental protocols.",
    "breast_cancer_3": "Updated recommendations focus on ESR1 mutation testing in HR-positive, HER2-negative metastatic breast cancer to guide treatment. The guideline supports elacestrant for ESR1-mutated cases post-endocrine therapy. Routine ESR1 testing in other cases is still not recommended.",
    "breast_cancer_4": "The guideline highlights sacituzumab govitecan for endocrine-resistant or HR-negative metastatic breast cancer based on the TROPiCS-02 trial. It suggests improved progression-free survival and overall survival with sacituzumab govitecan compared to standard chemotherapy. This is a direct update to prior recommendations.",
    "breast_cancer_5": "This guideline focuses on trastuzumab deruxtecan for HER2-low metastatic breast cancer, based on DESTINY-Breast04 data. It outlines efficacy in hormone receptor-positive patients and excludes HER2 IHC 0 tumors from this recommendation. It updates therapeutic options for HER2-low status.",
    "breast_cancer_6": "Updates systemic therapy biomarkers for metastatic breast cancer, emphasizing PIK3CA mutation testing for alpelisib eligibility and BRCA testing for PARP inhibitors. Other biomarkers like ESR1 and TROP2 are not yet supported for routine testing. This builds on earlier recommendations for precise molecular guidance.",
    "breast_cancer_7": "The update provides guidance on using biomarkers like Oncotype DX and MammaPrint to direct adjuvant chemotherapy and endocrine therapy in early-stage breast cancer. Recommendations vary based on menopausal status, nodal involvement, and genomic scores. HER2-positive and triple-negative cases are excluded from genomic test guidance.",
    "breast_cancer_8": "Recommends adjuvant bisphosphonates for postmenopausal breast cancer patients receiving systemic therapy to modestly improve overall survival. Denosumab is not recommended for recurrence prevention. This update incorporates additional data to refine earlier guidance.",
    "breast_cancer_9": "Supports pembrolizumab in combination with chemotherapy for stage II/III triple-negative breast cancer based on KEYNOTE-522 trial data. The guideline reports significant event-free survival improvements with this treatment approach. This update responds to practice-changing evidence.",
    "breast_cancer_10": "The guideline recommends abemaciclib combined with endocrine therapy for high-risk, HR-positive, HER2-negative, node-positive early breast cancer. It highlights improved invasive disease-free survival shown in the monarchE trial. This represents a focused update for adjuvant therapy options.",
    "breast_cancer_11": "The guideline recommends the use of adjuvant PARP inhibitor olaparib for patients with high-risk, early-stage HER2-negative breast cancer with germline BRCA1/2 mutations. It emphasizes significant improvement in invasive and distant disease-free survival shown in the OlympiA trial. This is a focused update addressing new clinical data for hereditary breast cancer management.",
    "breast_cancer_12": "This guideline update addresses optimal chemotherapy and targeted therapy for HER2-negative metastatic breast cancer. Recommendations include immune checkpoint inhibitors for PD-L1-positive triple-negative cases and PARP inhibitors for germline BRCA mutations. It emphasizes personalized approaches based on progression and genomic profiles.",
    "breast_cancer_13": "Updated recommendations include the use of alpelisib with endocrine therapy for PIK3CA-mutated, HR-positive, HER2-negative metastatic breast cancer. CDK4/6 inhibitors are emphasized for treatment-naïve HR-positive cases and those with progression on aromatase inhibitors. Routine ESR1 mutation testing is not supported due to insufficient data.",
    "breast_cancer_14": "This guideline focuses on axillary management in early-stage breast cancer, offering evidence-based recommendations for sentinel lymph node biopsy (SLNB) and further axillary interventions. It updates the ASCO 2017 guideline with considerations for neoadjuvant chemotherapy and radiotherapy. Recommendations highlight patient-centered approaches based on tumor location and clinical features.",
    "breast_cancer_15": "Guidance on neoadjuvant therapy emphasizes chemotherapy for triple-negative and HER2-positive breast cancer with high-risk or node-positive disease. Hormone therapy is suggested for HR-positive, HER2-negative postmenopausal cases. It also provides criteria for therapy selection based on tumor size and stage, avoiding routine treatment for very small, low-risk tumors.",
    "gu_cancer_1": "This guideline update focuses on the management of noncastrate advanced, recurrent, or metastatic prostate cancer. Triplet therapy using docetaxel plus androgen-deprivation therapy (ADT) combined with abiraterone or darolutamide is recommended for patients with high-volume disease. Updated results from major trials like ARASENS and PEACE-1 inform the recommendations for better survival outcomes.",
    "gu_cancer_2": "The guideline covers the management of metastatic clear cell renal cell carcinoma (ccRCC). First-line treatment depends on IMDC risk stratification, recommending combinations like immune checkpoint inhibitors and VEGFR TKIs. Cytoreductive nephrectomy is suggested for select cases, and second-line treatments include nivolumab or cabozantinib based on disease progression.",
    "gi_cancer_1": "Guidelines for systemic therapy in stage I-III anal squamous cell carcinoma recommend mitomycin-C with fluorouracil or capecitabine as radiosensitizing agents for chemoradiation. Cisplatin-based regimens are advised for immunosuppressed patients, but induction or post-CRT chemotherapy is discouraged. These recommendations refine therapeutic approaches based on clinical risk profiles.",
    "gi_cancer_2": "This guideline addresses immunotherapy and targeted therapies for advanced gastroesophageal cancer. It recommends nivolumab or pembrolizumab combined with chemotherapy based on HER2 status and PD-L1 scores. Trastuzumab-based therapies are preferred for HER2-positive cases, with evolving options highlighted for second-line treatments.",
    "gi_cancer_3": "The guideline provides recommendations for treating metastatic colorectal cancer, including first-line chemotherapy with VEGF or EGFR-targeted therapies based on molecular profiling. Encorafenib with cetuximab is suggested for BRAF-mutant cases, while cytoreductive surgery and systemic therapy are options for select patients. Treatment emphasizes multidisciplinary decision-making tailored to disease subtype.",
    "gi_cancer_4": "Adjuvant therapy for stage II colon cancer is not routinely recommended but is advised for high-risk subgroups like T4 tumors or those with lymphovascular invasion. Oxaliplatin-containing chemotherapy may be considered based on risk factors and shared decision-making. Updates incorporate recent findings on recurrence risk and treatment efficacy.",
    "headneck_cancer_1": "This guideline focuses on the prevention and management of osteoradionecrosis (ORN) in patients with head and neck cancer treated with radiation therapy. It emphasizes evidence-based recommendations for prevention prior to radiation therapy, surgical and nonsurgical management, and interdisciplinary coordination. Limited evidence supports hyperbaric oxygen, leukocyte-rich fibrin, or photobiomodulation, and these practices are not routinely recommended.",
    "headneck_cancer_2": "The guideline provides recommendations on immunotherapy and biomarker testing in recurrent or metastatic head and neck cancers. PD-L1 and tumor mutational burden (TMB) testing guide the selection of immune-checkpoint inhibitors, such as pembrolizumab and nivolumab. It includes first-line treatments and considers biomarkers for nasopharyngeal carcinoma.",
    "headneck_cancer_3": "This guideline covers management of salivary gland malignancies, emphasizing diagnosis and treatment tailored to histology and staging. Recommendations include preoperative imaging, surgical techniques, adjuvant radiotherapy for advanced disease, and systemic therapy for metastatic cases. Multidisciplinary tumor boards are encouraged for optimal patient-specific strategies.",
    "headneck_cancer_4": "Focuses on chemoradiotherapy for stage II-IVA nasopharyngeal carcinoma, highlighting the use of intensity-modulated radiotherapy (IMRT) and evidence-based chemotherapy sequences. Recommendations cover induction, concurrent, and adjuvant chemotherapy tailored to tumor stage and subtype. The guideline emphasizes precision in radiation planning and treatment.",
    "gu_cancer_1": "This guideline update addresses initial management of noncastrate advanced, recurrent, or metastatic prostate cancer. Triplet therapy with docetaxel, ADT, and either darolutamide or abiraterone is preferred for high-volume metastatic disease. Updated survival outcomes from trials like ARASENS and PEACE-1 underpin the recommendations.",
    "gu_cancer_2": "Guidelines for metastatic clear cell renal cell carcinoma (ccRCC) include systemic therapy options based on risk stratification using IMDC criteria. Cytoreductive nephrectomy is recommended for select patients with kidney-in-place and favorable or intermediate risk. First-line options include immune checkpoint inhibitors combined with VEGFR TKIs, while second-line therapies vary based on prior treatment.",
    "gynecologic_cancer_1": "This rapid update highlights the role of PARP inhibitors in ovarian cancer, focusing on rucaparib, olaparib, and niraparib for specific patient populations. Updated data emphasize risks in recurrent platinum-sensitive disease and potential survival detriments, leading to refined recommendations for maintenance therapy. The guideline incorporates recent FDA labeling changes and pivotal trial results.",
    "lung_cancer_1": "This update focuses on systemic therapy for stage IV non-small cell lung cancer (NSCLC) with actionable driver alterations. Recommendations include first-line osimertinib with chemotherapy for EGFR-mutated NSCLC and second-line amivantamab with chemotherapy for progression on third-generation tyrosine kinase inhibitors. The guideline reflects recent FDA approvals and clinical trial findings.",
    "lung_cancer_4": "Focuses on stage IV NSCLC without driver alterations, addressing immunotherapy combinations like nivolumab with ipilimumab for PD-L1-low cases. Recommendations adapt based on evolving evidence regarding treatment sequencing. The guideline is a continuation of earlier updates.",
    "lung_cancer_5": "This rapid recommendation update discusses adjuvant osimertinib for EGFR-positive stage I-IIIA NSCLC following resection. It emphasizes substantial disease-free survival benefits and includes atezolizumab for PD-L1 positive cases. This update is based on pivotal trials like Wu and Felip.",
    "lung_cancer_6": "This guideline covers the management of stage III NSCLC, emphasizing multimodal approaches. Recommendations include surgical evaluation, neoadjuvant/adjuvant therapy, and definitive chemoradiotherapy for unresectable cases. Tailored treatment plans are based on tumor characteristics and multidisciplinary input.",
    "neurooncology_cancer_1": "ASCO endorses ASTRO guidelines on radiation therapy for brain metastases, emphasizing stereotactic radiosurgery (SRS) for up to 10 metastases. Recommendations prioritize hippocampal-sparing WBRT and memantine to minimize cognitive decline. Endorsement adapts to new data on radiation techniques and outcomes.",
    "neurooncology_cancer_2": "This guideline focuses on managing brain metastases from solid tumors, advocating SRS for 1-4 metastases and local therapy for symptomatic cases. It outlines systemic therapy integration and highlights memantine for patients undergoing WBRT. Recommendations stress multidisciplinary care tailored to clinical scenarios.",
    "neurooncology_cancer_3": "Guidelines address therapy for diffuse astrocytic and oligodendroglial tumors, recommending PCV or TMZ-based regimens post-resection. Specific guidance is provided based on IDH mutation and 1p19q codeletion status. The guideline integrates molecular markers into treatment planning."
}