-- Safe example for the Database SQL Script uploader.
CREATE SCHEMA IF NOT EXISTS public;
CREATE TABLE public.teams (team_id INTEGER PRIMARY KEY, team_name VARCHAR NOT NULL);
CREATE TABLE public.employees (employee_id INTEGER PRIMARY KEY, employee_name VARCHAR NOT NULL, team_id INTEGER REFERENCES public.teams(team_id), salary DECIMAL(12,2));
INSERT INTO public.teams VALUES (1, 'Engineering'), (2, 'Finance');
INSERT INTO public.employees VALUES (101, 'John Smith', 1, 95000.00), (102, 'Sarah Johnson', 2, 85000.00);
