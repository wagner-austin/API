"""Seeding module for covenant-radar-api.

Provides functionality to populate the database with demo data for
testing and demonstration purposes.

Usage:
    from covenant_radar_api.seeding.runner import seed_database
    from covenant_radar_api.seeding.profiles_data_additional import ALL_PROFILES

    conn = psycopg.connect(dsn)
    result = seed_database(conn, ALL_PROFILES)

    # For synthetic data (hundreds of deals):
    from covenant_radar_api.seeding.synthetic import generate_synthetic_profiles

    profiles = generate_synthetic_profiles(n_deals=200, random_seed=42)
    result = seed_database(conn, profiles)
"""
