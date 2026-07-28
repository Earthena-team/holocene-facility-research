create schema if not exists supply_res_dev;

create table if not exists supply_res_dev.suppliers (
  id uuid primary key default gen_random_uuid(),
  name text not null,
  product text not null,
  normalized_key text not null unique,
  created_at timestamptz default now()
);

create table if not exists supply_res_dev.facilities (
  id uuid primary key default gen_random_uuid(),
  supplier_id uuid references supply_res_dev.suppliers(id) on delete cascade,
  facility_name text not null,
  facility_address text not null,
  latitude double precision,
  longitude double precision,
  facility_type text not null check (facility_type in ('manufacturing','logistics','raw_material')),
  confidence text not null,
  source_url text,
  geocode_status text not null default 'pending',
  created_at timestamptz default now(),
  unique (supplier_id, facility_name, facility_address)
);

create table if not exists supply_res_dev.research_runs (
  id uuid primary key,
  supplier_id uuid references supply_res_dev.suppliers(id),
  status text not null,
  input_tokens int,
  output_tokens int,
  thinking_tokens int,
  search_queries int,
  geocode_calls int,
  est_cost_usd numeric(10,5),
  error text,
  started_at timestamptz,
  finished_at timestamptz
);
