-- Patients treated by a doctor named James (example filter, not hardcoded application logic).
SELECT DISTINCT p.patient_id, p.first_name, p.last_name,
       d.first_name AS doctor_first_name, d.last_name AS doctor_last_name,
       dept.department_name
FROM public.patients AS p
JOIN public.appointments AS a ON a.patient_id = p.patient_id
JOIN public.doctors AS d ON d.doctor_id = a.doctor_id
JOIN public.departments AS dept ON dept.department_id = d.department_id
WHERE d.first_name = 'James'
ORDER BY p.last_name, p.first_name;

-- Appointments, diagnoses and prescriptions with medications.
-- LEFT JOIN retains appointments even when no diagnosis or prescription exists.
SELECT a.appointment_id, a.appointment_date,
       p.first_name AS patient_first_name, p.last_name AS patient_last_name,
       d.first_name AS doctor_first_name, d.last_name AS doctor_last_name,
       dept.department_name, dx.diagnosis_description,
       m.medication_name, rx.dosage
FROM public.appointments AS a
JOIN public.patients AS p ON p.patient_id = a.patient_id
JOIN public.doctors AS d ON d.doctor_id = a.doctor_id
JOIN public.departments AS dept ON dept.department_id = d.department_id
LEFT JOIN public.diagnoses AS dx ON dx.appointment_id = a.appointment_id
LEFT JOIN public.prescriptions AS rx ON rx.appointment_id = a.appointment_id
LEFT JOIN public.medications AS m ON m.medication_id = rx.medication_id
WHERE d.first_name = 'James'
ORDER BY a.appointment_date, a.appointment_id;
