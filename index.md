---
layout: default
title: Cody Maddox
---

# Cody Maddox

Graduate student in Analytics at Georgia Tech. I work on machine learning, clinical NLP, and computational data analysis.

---

<h2 id="projects">Projects</h2>

<div class="project-grid">
{% assign sorted = site.projects | sort: 'order' %}
{% for project in sorted %}
<div class="project-card">
  <h3><a href="{{ project.url | relative_url }}">{{ project.title }}</a></h3>
  <p>{{ project.summary }}</p>
  <div class="tags">
    {% for tag in project.tags %}
    <span class="tag">{{ tag }}</span>
    {% endfor %}
  </div>
</div>
{% endfor %}
</div>
