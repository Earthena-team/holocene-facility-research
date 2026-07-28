-- Add the product each facility was researched for. Backfill existing rows from
-- their supplier's product (facilities are supplier-scoped, and each supplier row
-- is one (name, product) pair), then enforce NOT NULL going forward.

alter table supply_res_dev.facilities
  add column if not exists product text;

update supply_res_dev.facilities f
  set product = s.product
  from supply_res_dev.suppliers s
  where f.supplier_id = s.id
    and f.product is null;

alter table supply_res_dev.facilities
  alter column product set default '';

update supply_res_dev.facilities
  set product = ''
  where product is null;

alter table supply_res_dev.facilities
  alter column product set not null;
