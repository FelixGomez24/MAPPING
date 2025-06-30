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

def hexbin_map_calls_rides_cr_improved():
    # Configuración de página
    st.set_page_config(layout="wide", page_title="Hexbin Map Analysis")
    st.title("Hexbin Map Analysis")
    
    # Función para guardar/recuperar configuraciones
    def save_config(config):
        if not os.path.exists('configs'):
            os.makedirs('configs')
        with open(f"configs/config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
            json.dump(config, f)
    
    def load_configs():
        if not os.path.exists('configs'):
            return []
        config_files = [f for f in os.listdir('configs') if f.endswith('.json')]
        configs = []
        for file in config_files:
            with open(f"configs/{file}", 'r') as f:
                configs.append((file, json.load(f)))
        return configs
    
    # Modo de comparación
    comparison_mode = st.checkbox("Enable comparison mode")
    
    if comparison_mode:
        st.sidebar.header("Comparison Mode")
        saved_configs = load_configs()
        
        if not saved_configs:
            st.warning("No saved configurations found. Save current config first.")
            comparison_mode = False
        else:
            selected_configs = st.sidebar.multiselect(
                "Select configurations to compare",
                [f"{config[0]} ({config[1]['city']} - {config[1]['date']})" for config in saved_configs],
                format_func=lambda x: x.split(' (')[0]
            )
            
            if selected_configs:
                st.header("Comparison View")
                cols = st.columns(len(selected_configs))
                
                for idx, config_ref in enumerate(selected_configs):
                    config_name = config_ref.split(' (')[0]
                    config = next(c[1] for c in saved_configs if c[0] == config_name)
                    
                    with cols[idx]:
                        st.subheader(f"{config['city']} - {config['date']}")
                        
                        # Mostrar mapa de Calls
                        fig_calls = go.Figure()
                        fig_calls.add_trace(go.Scattermapbox(
                            lat=[c[1] for c in config['center']],
                            lon=[c[0] for c in config['center']],
                            mode='markers',
                            marker=go.scattermapbox.Marker(size=9, color='red'),
                            text="Center Point",
                            hoverinfo='text'
                        ))
                        fig_calls.update_layout(
                            mapbox=dict(
                                style='carto-positron',
                                center={'lat': config['center'][0][1], 'lon': config['center'][0][0]},
                                zoom=10
                            ),
                            height=400,
                            margin=dict(l=10, r=10, t=30, b=10),
                            title="Calls Heatmap"
                        )
                        st.plotly_chart(fig_calls, use_container_width=True)
                        
                        # Mostrar estadísticas
                        st.metric("Total Calls", f"{config['total_calls']:,}")
                        st.metric("Total Rides", f"{config['total_rides']:,}")
                        st.metric("Global CR", f"{config['global_cr']:.1%}")
                        
                        # Botón para ver detalles
                        if st.button(f"View Details {idx+1}"):
                            st.session_state['current_config'] = config_name
                            st.experimental_rerun()
                
                return
    
    # File upload
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if not uploaded_file:
        st.warning("Please upload a CSV file")
        return
    
    # Read data
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return
    
    # Convert to numeric and clean
    numeric_cols = ['calls', 'ride', 'starting_lat', 'starting_lng', 'hr_call']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.fillna(0)
    
    # Date processing
    date_str = datetime.now().strftime('%d-%m-%Y')
    if 'stat_date' in df.columns:
        try:
            df['stat_date'] = pd.to_datetime(df['stat_date'], errors='coerce')
            valid_dates = df['stat_date'].dropna()
            if not valid_dates.empty:
                date_str = valid_dates.mode().iloc[0].strftime('%d-%m-%Y')
        except Exception as e:
            st.warning(f"Error processing dates: {e}. Using current date.")
    
    # Filter valid coordinates
    map_df = df[(df['starting_lat'] != 0) & (df['starting_lng'] != 0)].copy()
    if map_df.empty:
        st.error("No valid data after filtering coordinates.")
        return
    
    # City name
    city = map_df['City_Name'].iloc[0] if 'City_Name' in map_df.columns else 'City'
    
    # Sidebar filters
    with st.sidebar:
        st.header("Filters")
        
        # Hour filter
        hora_info = ""
        if 'hr_call' in map_df.columns:
            if st.checkbox("Filter by hour"):
                hora_input = st.text_input("Enter hour or range (e.g., 14 or 14-18):", "14")
                try:
                    registros_antes = len(map_df)
                    if '-' in hora_input:
                        hi, hf = map(int, hora_input.split('-'))
                        if hi <= hf:
                            mask = map_df['hr_call'].between(hi, hf)
                            hora_info = f"_hora_{hi}a{hf}"
                        else:
                            mask = (map_df['hr_call'] >= hi) | (map_df['hr_call'] <= hf)
                            hora_info = f"_hora_{hi}a{hf}"
                        map_df = map_df[mask]
                    else:
                        hora_especifica = int(hora_input)
                        mask = map_df['hr_call'] == hora_especifica
                        hora_info = f"_hora_{hora_especifica}"
                        map_df = map_df[mask]
                    st.write(f"Records after hour filter: {len(map_df)} (before: {registros_antes})")
                except ValueError:
                    st.warning("Invalid hour format. Hour filter will be ignored.")
        
        # Min calls filter
        min_calls_filter = None
        calls_info = ""
        if st.checkbox("Filter by minimum calls per hexagon"):
            min_calls = st.number_input("Minimum calls per hexagon:", min_value=1, value=10)
            if min_calls > 0:
                min_calls_filter = min_calls
                calls_info = f"_minCalls{min_calls}"
    
    # Set Mapbox token
    px.set_mapbox_access_token("pk.eyJ1Ijoiamx2b3J0dXphciIsImEiOiJjbGV3YzBlZXQwODc4M3dtemtkbHhvamI5In0.Kpmh1cwA9l9qz2K7n7iUkw")
    
    center = {'lat': map_df['starting_lat'].mean(), 'lon': map_df['starting_lng'].mean()}
    
    def create_hexbin_with_filter(data, col, title, scale, is_cr=False, min_calls_filter=None):
        zoom_level = 10
        if is_cr:
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
            tooltips = [f"📞 Calls: {c:.0f}<br>🚗 Rides: {r:.0f}<br>📈 CR: {cr:.1%}" if (min_calls_filter is None or c >= min_calls_filter) and not np.isnan(cr) else f"❌ Filtered ({c:.0f} calls < {min_calls_filter})" for c, r, cr in zip(zc, zr, zcr)]
            fig.data[0].text = tooltips
            fig.data[0].hovertemplate = "%{text}<extra></extra>"
            fig.update_layout(
                mapbox=dict(style='carto-positron', center=center, zoom=zoom_level),
                title=f"{title}",
                height=650,
                width=1200,
                margin=dict(l=10, r=10, t=50, b=10)
            )
            return fig, fc, fr, zc, zr, zcr
        else:
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
            tooltips = [f"{col.title()}: {v:.0f}" if not np.isnan(v) else "Filtered" for v in fig.data[0].z]
            fig.data[0].text = tooltips
            fig.data[0].hovertemplate = "%{text}<extra></extra>"
            fig.update_layout(
                mapbox=dict(style='carto-positron', center=center, zoom=zoom_level),
                title=f"{title}",
                height=650,
                width=1200,
                margin=dict(l=10, r=10, t=50, b=10)
            )
            return fig

    def extract_and_merge_cr_perimeters(hexbin_calls, hexbin_rides, zc, zr, zcr, min_calls_filter=None):
        try:
            from shapely.geometry import Polygon, MultiPolygon
            from shapely.ops import unary_union
            from shapely.geometry.polygon import orient
        except ImportError:
            st.warning("Shapely not installed. Perimeter extraction disabled.")
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
                st.warning("No valid hexagons found for merging.")
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
            st.error(f"Error merging perimeters: {str(e)}")
            return []

    # Create maps
    filtros_texto = calls_info + hora_info
    
    # Calculate global metrics
    total_calls = map_df['calls'].sum()
    total_rides = map_df['ride'].sum()
    global_cr = total_rides / total_calls if total_calls > 0 else 0
    
    # Save current configuration
    if st.sidebar.button("Save Current Configuration"):
        config = {
            'city': city,
            'date': date_str,
            'filters': filtros_texto,
            'center': [(center['lon'], center['lat'])],
            'total_calls': total_calls,
            'total_rides': total_rides,
            'global_cr': global_cr,
            'min_calls_filter': min_calls_filter
        }
        save_config(config)
        st.sidebar.success("Configuration saved!")
    
    # Create tabs
    tab1, tab2 = st.tabs(["📞 Calls Map", "📊 CR Map"])
    
    with tab1:
        container = st.container()
        with container:
            try:
                fig_calls = create_hexbin_with_filter(map_df, 'calls', f'Calls - {city} - {date_str}{filtros_texto}', 'bluered', False, min_calls_filter)
                st.plotly_chart(fig_calls, use_container_width=True)
                
                # Mostrar métricas globales
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Calls", f"{total_calls:,}")
                with col2:
                    st.metric("Total Rides", f"{total_rides:,}")
                with col3:
                    st.metric("Global CR", f"{global_cr:.1%}")
                
            except Exception as e:
                st.error(f"Error generating Calls map: {e}")
    
    with tab2:
        container = st.container()
        with container:
            try:
                cr_fig, fc, fr, zc, zr, zcr = create_hexbin_with_filter(map_df, None, f'CR - {city} - {date_str}{filtros_texto}', 'bluered', True, min_calls_filter)
                st.plotly_chart(cr_fig, use_container_width=True)
                
                # Generación y descarga de coordenadas fusionadas
                if st.button("Generate Merged CR Perimeters"):
                    with st.spinner("Merging adjacent hexagons..."):
                        perimeter_data = extract_and_merge_cr_perimeters(fc, fr, zc, zr, zcr, min_calls_filter)
                    
                    if perimeter_data:
                        # Crear contenido para descarga
                        coords_output_lines = []
                        for area in perimeter_data:
                            area_header = f"# {area['area_id']} | CR Range: {area['cr_range']} | Hexagons: {area['num_hexagons']} | Avg CR: {area['avg_cr']:.3f}"
                            coords = "\n".join([f"{lon:.6f},{lat:.6f}" for lon, lat in area['coordinates']])
                            coords_output_lines.append(f"{area_header}\n{coords}")
                        
                        coords_text = "\n\n".join(coords_output_lines)
                        
                        # Botón de descarga
                        st.download_button(
                            label="⬇️ Download Merged Perimeters",
                            data=coords_text,
                            file_name=f"merged_perimeters_{city}_{date_str}{filtros_texto}.txt",
                            mime="text/plain"
                        )
                        
                        # Resumen estadístico
                        st.success(f"Successfully generated {len(perimeter_data)} merged areas:")
                        st.write(f"- CR Low: {sum(1 for p in perimeter_data if p['cr_range'] == 'CR_Low')}")
                        st.write(f"- CR Medium: {sum(1 for p in perimeter_data if p['cr_range'] == 'CR_Medium')}")
                        st.write(f"- CR High: {sum(1 for p in perimeter_data if p['cr_range'] == 'CR_High')}")
                        st.write(f"- Total hexagons included: {sum(p['num_hexagons'] for p in perimeter_data)}")
                    else:
                        st.warning("No merged areas were generated. Try adjusting your filters.")
            except Exception as e:
                st.error(f"Error processing CR data: {e}")

if __name__ == "__main__":
    hexbin_map_calls_rides_cr_improved()