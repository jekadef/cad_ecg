SELECT msdw_2020_phi.d_person.MEDICAL_RECORD_NUMBER,
    msdw_2020_phi.d_person.PERSON_KEY,
    msdw_2020_phi.d_person.GENDER,
    msdw_2020_phi.d_person.RACE,
    msdw_2020_phi.d_person.PATIENT_ETHNIC_GROUP,
    msdw_2020_phi.d_person.DATE_OF_BIRTH,
    msdw_2020_phi.fd_diagnosis.CONTEXT_DIAGNOSIS_CODE,
    msdw_2020_phi.d_calendar.CALENDAR_DATE

FROM msdw_2020_phi.d_person
INNER JOIN msdw_2020_phi.fact_d ON msdw_2020_phi.d_person.PERSON_KEY = msdw_2020_phi.fact_d.PERSON_KEY
INNER JOIN msdw_2020_phi.b_diagnosis ON msdw_2020_phi.fact_d.DIAGNOSIS_GROUP_KEY = msdw_2020_phi.b_diagnosis.DIAGNOSIS_GROUP_KEY
INNER JOIN msdw_2020_phi.fd_diagnosis ON msdw_2020_phi.b_diagnosis.DIAGNOSIS_KEY = msdw_2020_phi.fd_diagnosis.DIAGNOSIS_KEY
INNER JOIN msdw_2020_phi.d_calendar ON msdw_2020_phi.fact_d.CALENDAR_KEY = msdw_2020_phi.d_calendar.CALENDAR_KEY
WHERE msdw_2020_phi.fd_diagnosis.CONTEXT_DIAGNOSIS_CODE LIKE 'I%'
AND msdw_2020_phi.fd_diagnosis.CONTEXT_NAME = "ICD-10"

