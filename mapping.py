import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from datetime import datetime
import math
import json
import os
from io import BytesIO

def hexbin_map_calls_rides_cr_improved():
    st.set_page_config(layout="wide", page_title="Map Analysis", page_icon="mapping/map.png")
    st.title("Hexbin Map Analysis")
    
    # Función para fusionar hexágonos basada en rangos de CR
    def extract_and_merge_cr_perimeters(hexbin_calls, hexbin_rides, zc, zr, zcr, min_calls_filter=None):
        try:
            from shapely.geometry import Polygon, MultiPolygon
            from shapely.ops import unary_union
            from shapely.geometry.polygon import orient
        except ImportError:
            st.error("❌ Shapely library is required for this feature. Please install it with: pip install shapely")
            return []

        try:
            hexbin_trace = hexbin_calls.data[0]
            hex_polygons = []
            cr_values = []
            
            for i in range(len(zc)):
                if (min_calls_filter is None or zc[i] >= min_calls_filter) and not np.isnan(zcr[i]) and zcr[i] > 0:
                    cr_value = zcr[i]
                    if 0.0 <= cr_value < 0.3:
                        cr_range = "CR_Low"
                    elif 0.3 <= cr_value < 0.6:
                        cr_range = "CR_Medium"
                    elif 0.6 <= cr_value <= 1.0:
                        cr_range = "CR_High"
                    else:
                        continue
                    
                    if hasattr(hexbin_trace, 'geojson') and hexbin_trace.geojson and i < len(hexbin_trace.geojson['features']):
                        coords = hexbin_trace.geojson['features'][i]['geometry']['coordinates'][0]
                    else:
                        lat_center = hexbin_trace.lat[i] if hasattr(hexbin_trace, 'lat') else hexbin_trace.y[i]
                        lon_center = hexbin_trace.lon[i] if hasattr(hexbin_trace, 'lon') else hexbin_trace.x[i]
                        hex_radius = 0.005
                        vertices = []
                        for angle in range(0, 360, 60):
                            angle_rad = math.radians(angle)
                            lat_offset = hex_radius * math.cos(angle_rad)
                            lon_offset = hex_radius * math.sin(angle_rad) / math.cos(math.radians(lat_center))
                            vertex_lat = lat_center + lat_offset
                            vertex_lon = lon_center + lon_offset
                            vertices.append((vertex_lon, vertex_lat))
                        coords = vertices
                    
                    polygon = Polygon(coords)
                    hex_polygons.append(polygon)
                    cr_values.append(cr_range)
            
            if not hex_polygons:
                st.warning("⚠️ No valid hexagons found for merging. Check your filters.")
                return []

            cr_groups = {}
            for polygon, cr_range in zip(hex_polygons, cr_values):
                if cr_range not in cr_groups:
                    cr_groups[cr_range] = []
                cr_groups[cr_range].append(polygon)
            
            merged_perimeters = []
            for cr_range, polygons in cr_groups.items():
                merged = unary_union(polygons)
                
                if isinstance(merged, Polygon):
                    merged = [merged]
                elif isinstance(merged, MultiPolygon):
                    merged = list(merged.geoms)
                
                for i, polygon in enumerate(merged):
                    oriented = orient(polygon, sign=1.0)
                    coords = list(oriented.exterior.coords)
                    merged_perimeters.append({
                        'area_id': f"{cr_range}_{i+1}",
                        'cr_range': cr_range,
                        'coordinates': coords,
                        'num_hexagons': sum(p.within(polygon) or p.touches(polygon) for p in polygons),
                        'avg_cr': np.mean([zcr[i] for i in range(len(zcr)) 
                                 if (min_calls_filter is None or zc[i] >= min_calls_filter) 
                                 and not np.isnan(zcr[i]) 
                                 and ((cr_range == "CR_Low" and 0.0 <= zcr[i] < 0.3) or
                                      (cr_range == "CR_Medium" and 0.3 <= zcr[i] < 0.6) or
                                      (cr_range == "CR_High" and 0.6 <= zcr[i] <= 1.0))])
                    })
            
            return merged_perimeters
        
        except Exception as e:
            st.error(f"❌ Error merging perimeters: {str(e)}")
            if "NoneType" in str(e):
                st.info("This error might occur if there's no data to merge. Try adjusting your filters.")
            return []
    
    uploaded_file = st.file_uploader("Upload data file", type=["csv", "xlsx", "xls"])
    if not uploaded_file:
        st.info("Please upload a CSV or Excel file to begin analysis")
        return
    
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
            
        required_columns = {'calls', 'ride', 'starting_lat', 'starting_lng'}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            st.error(f"❌ Missing required columns in file: {', '.join(missing_columns)}")
            st.info("The file must contain at least these columns: calls, ride, starting_lat, starting_lng")
            return
            
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        st.info("Please check that the file is not corrupted and is in the correct format")
        return
    
    numeric_cols = ['calls', 'ride', 'starting_lat', 'starting_lng', 'hr_call']
    conversion_errors = []
    
    for c in numeric_cols:
        if c in df.columns:
            original_non_null = df[c].notnull().sum()
            df[c] = pd.to_numeric(df[c], errors='coerce')
            new_non_null = df[c].notnull().sum()
            if new_non_null < original_non_null:
                conversion_errors.append(f"{c}: {original_non_null - new_non_null} values converted to NaN")
    
    if conversion_errors:
        st.warning("Some values couldn't be converted to numbers:\n" + "\n".join(conversion_errors))
    
    df = df.fillna(0)
    
    date_str = datetime.now().strftime('%d-%m-%Y')
    if 'stat_date' in df.columns:
        try:
            df['stat_date'] = pd.to_datetime(df['stat_date'], errors='coerce')
            valid_dates = df['stat_date'].dropna()
            if not valid_dates.empty:
                date_str = valid_dates.mode().iloc[0].strftime('%d-%m-%Y')
            else:
                st.warning("Date column exists but no valid dates found. Using current date.")
        except Exception as e:
            st.warning(f"Error processing dates: {e}. Using current date.")
    
    initial_count = len(df)
    map_df = df[(df['starting_lat'] != 0) & (df['starting_lng'] != 0)].copy()
    
    if map_df.empty:
        st.error("❌ No valid data after filtering coordinates (lat/lng != 0)")
        st.info("Check that your starting_lat and starting_lng columns contain valid coordinates")
        return
    
    removed_count = initial_count - len(map_df)
    if removed_count > 0:
        st.warning(f"⚠️ Removed {removed_count} records with invalid coordinates (lat/lng = 0)")
    
    city = map_df['City_Name'].iloc[0] if 'City_Name' in map_df.columns else 'City'
    
    with st.sidebar:
        st.header("Filters")
        
        hora_info = ""
        if 'hr_call' in map_df.columns:
            unique_hours = sorted(map_df['hr_call'].unique())
            if st.checkbox("Filter by hour", key='hour_filter'):
                hora_input = st.text_input("Enter hour or range (e.g., 14 or 14-18):", "14")
                
                try:
                    registros_antes = len(map_df)
                    
                    if '-' in hora_input:
                        hi, hf = map(int, hora_input.split('-'))
                        if hi < 0 or hi > 23 or hf < 0 or hf > 23:
                            st.error("❌ Hours must be between 0 and 23")
                        elif hi > hf:
                            st.error("❌ Start hour must be before end hour")
                        else:
                            mask = map_df['hr_call'].between(hi, hf)
                            hora_info = f"_hora_{hi}a{hf}"
                            map_df = map_df[mask]
                    else:
                        hora_especifica = int(hora_input)
                        if hora_especifica < 0 or hora_especifica > 23:
                            st.error("❌ Hour must be between 0 and 23")
                        else:
                            mask = map_df['hr_call'] == hora_especifica
                            hora_info = f"_hora_{hora_especifica}"
                            map_df = map_df[mask]
                    
                    if len(map_df) == 0:
                        st.error(f"❌ No data available for selected hours. Available hours: {unique_hours}")
                        map_df = df[(df['starting_lat'] != 0) & (df['starting_lng'] != 0)].copy()
                    else:
                        st.info(f"Records after hour filter: {len(map_df)} (before: {registros_antes})")
                except ValueError:
                    st.error("❌ Invalid hour format. Please use numbers between 0-23 or range like 14-18")
        
        min_calls_filter = None
        calls_info = ""
        if st.checkbox("Filter by minimum calls per hexagon", key='min_calls_filter'):
            min_calls = st.number_input("Minimum calls per hexagon:", min_value=1, value=10)
            if min_calls > 0:
                min_calls_filter = min_calls
                calls_info = f"_minCalls{min_calls}"
                st.info(f"Will filter hexagons with less than {min_calls} calls")
    
    px.set_mapbox_access_token("pk.eyJ1Ijoiamx2b3J0dXphciIsImEiOiJjbGV3YzBlZXQwODc4M3dtemtkbHhvamI5In0.Kpmh1cwA9l9qz2K7n7iUkw")
    
    center = {'lat': map_df['starting_lat'].mean(), 'lon': map_df['starting_lng'].mean()}
    
    # Función para crear mapas de hexbin con o sin cálculo de CR
    def create_hexbin_with_filter(data, col, title, scale, is_cr=False, min_calls_filter=None):
        zoom_level = 10
        if is_cr:
            try:
                fc = ff.create_hexbin_mapbox(
                    data_frame=data, lat='starting_lat', lon='starting_lng',
                    nx_hexagon=30, opacity=0.4, labels={'color': 'calls'},
                    min_count=1, color='calls', agg_func=np.sum,
                    show_original_data=False, center=center, zoom=zoom_level
                )
                fr = ff.create_hexbin_mapbox(
                    data_frame=data, lat='starting_lat', lon='starting_lng',
                    nx_hexagon=30, opacity=0.4, labels={'color': 'ride'},
                    min_count=1, color='ride', agg_func=np.sum,
                    show_original_data=False, center=center, zoom=zoom_level
                )
                zc, zr = fc.data[0].z, fr.data[0].z
                
                if min_calls_filter is not None:
                    valid_mask = zc >= min_calls_filter
                    zc_filtered = np.where(valid_mask, zc, np.nan)
                    zr_filtered = np.where(valid_mask, zr, np.nan)
                else:
                    zc_filtered, zr_filtered = zc, zr
                
                zcr = np.where((zc_filtered > 0) & ~np.isnan(zc_filtered), zr_filtered / zc_filtered, np.nan)
                fig = go.Figure(fc.data[0])
                fig.data[0].z = zcr
                fig.data[0].colorscale = scale
                fig.data[0].zmin, fig.data[0].zmax = 0, 1
                fig.data[0].colorbar.title = 'CR'
                
                tooltips = []
                for c, r, cr in zip(zc, zr, zcr):
                    if (min_calls_filter is None or c >= min_calls_filter) and not np.isnan(cr):
                        tooltips.append(f"📞 Calls: {c:.0f}<br>🚗 Rides: {r:.0f}<br>📈 CR: {cr:.1%}")
                    elif min_calls_filter is not None and c < min_calls_filter:
                        tooltips.append(f"❌ Filtered (only {c:.0f} calls)")
                    else:
                        tooltips.append("No data")
                
                fig.data[0].text = tooltips
                fig.data[0].hovertemplate = "%{text}<extra></extra>"
                fig.update_layout(
                    mapbox=dict(style='carto-positron', center=center, zoom=zoom_level),
                    title=f"CR - {city} - {date_str}{filtros_texto}",
                    height=650,
                    width=1200,
                    margin=dict(l=10, r=10, t=50, b=10)
                )
                return fig, fc, fr, zc, zr, zcr
            except Exception as e:
                st.error(f"❌ Error creating CR map: {str(e)}")
                raise
        else:
            try:
                fig = ff.create_hexbin_mapbox(
                    data_frame=data, lat='starting_lat', lon='starting_lng',
                    nx_hexagon=30, opacity=0.4, labels={'color': col},
                    min_count=1, color=col, agg_func=np.sum,
                    show_original_data=False, color_continuous_scale=scale,
                    center=center, zoom=zoom_level
                )
                
                if min_calls_filter is not None and col == 'calls':
                    z_values = fig.data[0].z
                    filtered_z = np.where(z_values >= min_calls_filter, z_values, np.nan)
                    fig.data[0].z = filtered_z
                
                tooltips = []
                for v in fig.data[0].z:
                    if not np.isnan(v):
                        tooltips.append(f"{col.title()}: {v:.0f}")
                    else:
                        tooltips.append("Filtered (min calls not met)" if min_calls_filter else "No data")
                
                fig.data[0].text = tooltips
                fig.data[0].hovertemplate = "%{text}<extra></extra>"
                fig.update_layout(
                    mapbox=dict(style='carto-positron', center=center, zoom=zoom_level),
                    title=f"Calls - {city} - {date_str}{filtros_texto}",
                    height=650,
                    width=1200,
                    margin=dict(l=10, r=10, t=50, b=10)
                )
                return fig
            except Exception as e:
                st.error(f"❌ Error creating {col} map: {str(e)}")
                raise

    filtros_texto = calls_info + hora_info
    
    total_calls = map_df['calls'].sum()
    total_rides = map_df['ride'].sum()
    global_cr = total_rides / total_calls if total_calls > 0 else 0
    
    tab1, tab2 = st.tabs(["📞 Calls Map", "📊 CR Map"])
    
    with tab1:
        container = st.container()
        with container:
            try:
                fig_calls = create_hexbin_with_filter(map_df, 'calls', f'Calls - {city} - {date_str}{filtros_texto}', 'bluered', False, min_calls_filter)
                st.plotly_chart(fig_calls, use_container_width=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Calls", f"{total_calls:,}")
                with col2:
                    st.metric("Total Rides", f"{total_rides:,}")
                with col3:
                    st.metric("Global CR", f"{global_cr:.1%}")
                
            except Exception as e:
                st.error(f"❌ Error generating Calls map: {str(e)}")
                if "negative dimensions" in str(e):
                    st.info("This error often occurs when there's not enough data after filtering. Try adjusting your filters.")
    
    with tab2:
        container = st.container()
        with container:
            try:
                cr_fig, fc, fr, zc, zr, zcr = create_hexbin_with_filter(map_df, None, f'CR - {city} - {date_str}{filtros_texto}', 'bluered', True, min_calls_filter)
                st.plotly_chart(cr_fig, use_container_width=True)
                
                if st.button("Generate Merged CR Perimeters"):
                    with st.spinner("Merging adjacent hexagons..."):
                        perimeter_data = extract_and_merge_cr_perimeters(fc, fr, zc, zr, zcr, min_calls_filter)
                    
                    if perimeter_data:
                        coords_output_lines = []
                        for area in perimeter_data:
                            area_header = f"# {area['area_id']} | CR Range: {area['cr_range']} | Hexagons: {area['num_hexagons']} | Avg CR: {area['avg_cr']:.3f}"
                            coords = "\n".join([f"{lon:.6f},{lat:.6f}" for lon, lat in area['coordinates']])
                            coords_output_lines.append(f"{area_header}\n{coords}")
                        
                        coords_text = "\n\n".join(coords_output_lines)
                        
                        st.download_button(
                            label="⬇️ Download Merged Perimeters",
                            data=coords_text,
                            file_name=f"merged_perimeters_{city}_{date_str}{filtros_texto}.txt",
                            mime="text/plain"
                        )
                        
                        st.success(f"Successfully generated {len(perimeter_data)} merged areas:")
                        st.write(f"- CR Low: {sum(1 for p in perimeter_data if p['cr_range'] == 'CR_Low')}")
                        st.write(f"- CR Medium: {sum(1 for p in perimeter_data if p['cr_range'] == 'CR_Medium')}")
                        st.write(f"- CR High: {sum(1 for p in perimeter_data if p['cr_range'] == 'CR_High')}")
                        st.write(f"- Total hexagons included: {sum(p['num_hexagons'] for p in perimeter_data)}")
                    else:
                        st.warning("⚠️ No merged areas were generated. Try adjusting your filters or minimum calls threshold.")
            except Exception as e:
                st.error(f"❌ Error processing CR data: {str(e)}")
                if "negative dimensions" in str(e):
                    st.info("This error often occurs when there's not enough data after filtering. Try adjusting your filters.")

if __name__ == "__main__":
    hexbin_map_calls_rides_cr_improved()