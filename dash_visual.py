import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go

# Load data
df = pd.read_csv(r'train.csv')
corr_df = pd.read_csv(r'data\df_target_correlation.csv')
corr_df['correlation'] = abs(corr_df['correlation'])
corr_df = corr_df[corr_df['feature_2'] != 'critical_temp']
corr_df_cat = pd.read_csv(r'data\cat_correlation_strength.csv')

for col in df.columns:
    unique_vals = df[col].dropna().unique()
    if len(unique_vals) <= 2:
        df[col] = df[col].astype('category')

#numerical_cols = ['demographics_age_index_ecg','ecg_resting_hr','ecg_resting_pr','ecg_resting_qrs','ecg_resting_qtc','hgb_peri','hct_peri','rdw_peri','wbc_peri','plt_peri','inr_peri','ptt_peri','esr_peri','crp_high_sensitive_peri','albumin_peri','alkaline_phophatase_peri','alanine_transaminase_peri','aspartate_transaminase_peri','bilirubin_total_peri']
df_numerical = df.select_dtypes(include=['number'])
df_numerical['critical_temp'] = df['critical_temp']
df_categorical = df.select_dtypes(include=['object','category','bool'])

# Dash app
app = dash.Dash(__name__)
app.title = "Interactive Exploratory Data Analysis"

app.layout = html.Div([
    html.H2("Exploratory Data Analysis", style={'textAlign': 'center'}),

    # Bar plot and PCA side by side
    html.Div([
        html.Div([
            html.H3("Correlation of Features with Critical temperature", style ={'textAlign': 'center'}),
            dcc.Graph(
                id='correlation-bar',
                figure=px.bar(
                    data_frame=corr_df[corr_df['feature_2'].isin(df_numerical.columns)].head(10).sort_values(by= 'correlation', ascending=True),
                    y='feature_2',
                    x='correlation',
                    hover_data='correlation',
                    color='correlation',
                    color_continuous_scale=['#add8e6', '#87ceeb', '#1e90ff', '#4169e1'],
                    labels={'correlation': 'Absolute Correlation Value', 'feature_2': 'Top ten highly correlated features'},
                    width=900,
                    height=700
                ).update_layout(coloraxis_showscale=False,  plot_bgcolor='white', xaxis=dict(showgrid=True, gridcolor='lightgray'), yaxis=dict(showgrid=True, gridcolor='lightgray'))
            )
        ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top', 'height': '700px'}),

        html.Div([
            html.Div(id='selected-feature'),
            html.Div(
                dcc.Graph(id='box-plot'),
                style={'height': '400px'}
            ),
            html.Div(
                dcc.Graph(id='histogram'),
                style={'height': '300px'}
            )
        ], style={'width': '45%', 'display': 'inline-block', 'verticalAlign': 'top', 'paddingLeft': '20px'})
        ]),   
])

@app.callback(
    Output('box-plot', 'figure'),
    Output('histogram', 'figure'),
    Output('selected-feature', 'children'),
    Input('correlation-bar', 'hoverData')
)
def update_boxplot_on_hover(num_hover):
    ctx = dash.callback_context
    empty_fig = go.Figure()
    empty_fig.add_annotation(
        text = "Box Plot not available for categorical columns", 
        showarrow=False
    )
    empty_fig.update_layout(xaxis={'visible': False}, yaxis={'visible': False}, plot_bgcolor='white')

    if not ctx.triggered:
        return {},{}, "Hover over a bar to see the distribution"

    prop_id = ctx.triggered[0]['prop_id']
    if prop_id == 'correlation-bar.hoverData' and num_hover:
        hoverData = num_hover
        hovered_feature = hoverData['points'][0]['y']
        box_fig = px.box(
            data_frame=df_numerical,
            x=hovered_feature,
            orientation='h'
        )
        box_fig.update_layout(width=800,  plot_bgcolor='white', title={'text': f'Distribution of {hovered_feature} by Target','x': 0.5,'xanchor': 'center'},  xaxis=dict(showgrid=True, gridcolor='lightgray'), yaxis=dict(showgrid=True, gridcolor='lightgray')),

        hist_fig = px.histogram(
            data_frame=df,
            x=hovered_feature,
            title=f"Histogram of {hovered_feature}",
            nbins=50
        )
        hist_fig.update_layout(width=800, plot_bgcolor='white', title={'text': f'Histogram of {hovered_feature}','x': 0.5,'xanchor': 'center'},  xaxis=dict(showgrid=True, gridcolor='lightgray'), yaxis=dict(showgrid=True, gridcolor='lightgray'))

        return box_fig, hist_fig, f"Selected Feature: {hovered_feature}"
    

    else:
        return {},{},"Hover over a bar to see the distribution"


# Run the app
if __name__ == '__main__':
    app.run(debug=True)