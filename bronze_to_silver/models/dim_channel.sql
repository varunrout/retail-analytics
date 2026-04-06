-- dim_channel: Sales channel dimension
-- Grain: one row per channel
-- Static seed table

SELECT
  channel_key,
  channel_id,
  channel_name,
  channel_type,
  marketplace_fee_pct,
  is_direct,
  is_active
FROM (
  VALUES
    (1, 'shopify_uk',   'Shopify UK Store',  'DTC',         2.90,  TRUE,  TRUE),
    (2, 'ebay_uk',      'eBay UK',           'Marketplace', 12.80, FALSE, TRUE),
    (3, 'amazon_uk',    'Amazon UK',         'Marketplace', 15.45, FALSE, TRUE),
    (4, 'store_uk',     'Retail Store',      'Physical',    0.00,  TRUE,  TRUE),
    (5, 'wholesale_uk', 'Wholesale',         'B2B',         0.00,  TRUE,  FALSE)
) AS t(channel_key, channel_id, channel_name, channel_type, marketplace_fee_pct, is_direct, is_active)
