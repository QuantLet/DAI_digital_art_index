### import packages 
import matplotlib.pyplot as plt

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.graphics.gofplots import qqplot

### plotting function 
def qqploting(resid, axis_range):
    
    qqplot_data = qqplot(resid, line='s').gca().lines
    plt.close()
    
    fig = make_subplots(specs=[[{"secondary_y": False}]])
    fig['layout'].update(height=800, width=800,
                         title='',
                         showlegend=False,
                         font=dict(family='Times New Roman', size=32))

    fig.add_trace(go.Scatter(
        y=qqplot_data[0].get_ydata(),
        x=qqplot_data[0].get_xdata(),
        mode='markers', 
        marker=dict(color='white', line=dict(color='black', width=1))  
        ), secondary_y=False)

    # fig.add_trace(go.Scatter(
    #     y=qqplot_data[1].get_ydata(),
    #     x=qqplot_data[1].get_xdata(),
    #     line=dict(color='red', dash='solid', width=2)
    #     ), secondary_y=False)

    fig.add_shape(type='line',
                yref='paper', xref='paper',
                y0=0, y1=1, x0=0, x1=1,
                line=dict(color='blue'))

    fig['layout']['xaxis'].update(title='Theoretical quantiles')
    fig['layout']['yaxis'].update(title='Sample quantiles')

    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True,
                    showgrid=False, range=axis_range)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True, 
                    showgrid=False, range=axis_range)


    fig.update_layout({'plot_bgcolor': 'rgba(0,0,0,0)',
                    'paper_bgcolor': 'rgba(0,0,0,0)'},
                    font_color='black')

    return(fig)


