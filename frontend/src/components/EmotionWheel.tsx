import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';

interface EmotionData {
  emotion: string;
  value: number;
  color: string;
  icon: string;
}

interface Props {
  emotions: EmotionData[];
  size?: number;
  onEmotionClick?: (emotion: string) => void;
}

const EmotionWheel: React.FC<Props> = ({ emotions, size = 400, onEmotionClick }) => {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || !emotions.length) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const width = size;
    const height = size;
    const radius = Math.min(width, height) / 2 - 40;

    const g = svg
      .append('g')
      .attr('transform', `translate(${width / 2},${height / 2})`);

    // Échelle radiale
    const angleScale = d3
      .scaleLinear()
      .domain([0, emotions.length])
      .range([0, 2 * Math.PI]);

    // Échelle de rayon basée sur la valeur
    const radiusScale = d3
      .scaleLinear()
      .domain([0, 1])
      .range([radius * 0.3, radius]);

    // Création des arcs
    emotions.forEach((emotion, i) => {
      const startAngle = angleScale(i);
      const endAngle = angleScale(i + 1);
      const innerRadius = radius * 0.3;
      const outerRadius = radiusScale(emotion.value);

      const arc = d3
        .arc<any>()
        .innerRadius(innerRadius)
        .outerRadius(outerRadius)
        .startAngle(startAngle)
        .endAngle(endAngle);

      // Groupe pour chaque émotion
      const emotionGroup = g
        .append('g')
        .attr('class', 'emotion-group')
        .style('cursor', 'pointer')
        .on('click', () => onEmotionClick?.(emotion.emotion));

      // Arc principal
      emotionGroup
        .append('path')
        .attr('d', arc)
        .attr('fill', emotion.color)
        .attr('opacity', 0.8)
        .on('mouseover', function() {
          d3.select(this)
            .transition()
            .duration(200)
            .attr('opacity', 1)
            .attr('transform', 'scale(1.05)');
        })
        .on('mouseout', function() {
          d3.select(this)
            .transition()
            .duration(200)
            .attr('opacity', 0.8)
            .attr('transform', 'scale(1)');
        });

      // Texte de l'émotion
      const labelAngle = (startAngle + endAngle) / 2;
      const labelRadius = (innerRadius + outerRadius) / 2;
      const x = Math.cos(labelAngle - Math.PI / 2) * labelRadius;
      const y = Math.sin(labelAngle - Math.PI / 2) * labelRadius;

      emotionGroup
        .append('text')
        .attr('x', x)
        .attr('y', y - 10)
        .attr('text-anchor', 'middle')
        .attr('font-size', '24px')
        .text(emotion.icon);

      emotionGroup
        .append('text')
        .attr('x', x)
        .attr('y', y + 10)
        .attr('text-anchor', 'middle')
        .attr('font-size', '12px')
        .attr('fill', 'white')
        .attr('font-weight', 'bold')
        .text(emotion.emotion);

      // Valeur en pourcentage
      emotionGroup
        .append('text')
        .attr('x', x)
        .attr('y', y + 25)
        .attr('text-anchor', 'middle')
        .attr('font-size', '10px')
        .attr('fill', 'white')
        .text(`${Math.round(emotion.value * 100)}%`);
    });

    // Centre avec le texte "Émotions"
    g.append('circle')
      .attr('r', radius * 0.25)
      .attr('fill', '#1e1e1e')
      .attr('opacity', 0.9);

    g.append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', '0.35em')
      .attr('fill', 'white')
      .attr('font-size', '16px')
      .attr('font-weight', 'bold')
      .text('Émotions');

  }, [emotions, size, onEmotionClick]);

  return (
    <div className="emotion-wheel-container">
      <svg ref={svgRef} width={size} height={size} />
    </div>
  );
};

export default EmotionWheel;