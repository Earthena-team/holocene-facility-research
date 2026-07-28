-- Normalize supplier identity: one row per COMPANY, not per (company, product).
-- A supplier can supply many products; those products are represented by the
-- distinct `product` values on its facilities and research_runs. The per-(supplier,
-- product) cache/dedupe invariant is unchanged — it just keys on normalized_name +
-- product now instead of a fused normalized_key.
--
-- Safe to run once on the existing schema. Collapses any duplicate (name, product)
-- supplier rows that share a company name into a single canonical supplier row,
-- repointing their facilities and research_runs first.

set search_path to supply_res_dev;

-- 1. research_runs gains `product` (which product the run was for). Backfill from
--    the run's current supplier row BEFORE we drop suppliers.product below.
alter table research_runs add column if not exists product text;
update research_runs r
  set product = s.product
  from suppliers s
  where r.supplier_id = s.id and r.product is null;
alter table research_runs alter column product set default '';
update research_runs set product = '' where product is null;
alter table research_runs alter column product set not null;

-- 2. suppliers gains normalized_name (company identity). Backfill with an ASCII slug
--    approximation of python-slugify(lower(name)); unicode edge cases may differ and
--    simply cause a one-time cache miss (re-research), never data loss.
alter table suppliers add column if not exists normalized_name text;
update suppliers
  set normalized_name = trim(both '-' from regexp_replace(lower(name), '[^a-z0-9]+', '-', 'g'))
  where normalized_name is null;

-- 3. Collapse duplicate supplier rows (same normalized_name). Keep the earliest
--    row as canonical, repoint children, delete the rest.
create temporary table _supplier_remap on commit drop as
select s.id as old_id, k.id as new_id
from suppliers s
join (
  select distinct on (normalized_name) id, normalized_name
  from suppliers
  order by normalized_name, created_at, id
) k on k.normalized_name = s.normalized_name
where s.id <> k.id;

update facilities f
  set supplier_id = m.new_id
  from _supplier_remap m
  where f.supplier_id = m.old_id;

update research_runs r
  set supplier_id = m.new_id
  from _supplier_remap m
  where r.supplier_id = m.old_id;

delete from suppliers s
  using _supplier_remap m
  where s.id = m.old_id;

-- 4. Swap supplier identity constraints: drop (name, product) key, enforce one
--    row per company.
alter table suppliers drop constraint if exists suppliers_normalized_key_key;
alter table suppliers drop column if exists normalized_key;
alter table suppliers drop column if exists product;
alter table suppliers alter column normalized_name set not null;
create unique index if not exists suppliers_normalized_name_key
  on suppliers (normalized_name);

-- 5. A facility is now unique per (supplier, product, name, address) — the same
--    physical plant can appear under two products for one supplier.
alter table facilities
  drop constraint if exists facilities_supplier_id_facility_name_facility_address_key;
create unique index if not exists facilities_supplier_product_name_address_key
  on facilities (supplier_id, product, facility_name, facility_address);
