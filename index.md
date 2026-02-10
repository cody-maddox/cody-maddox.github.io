---
layout: default
title: Cody Maddox
---

# Cody Maddox

Healthcare professional with 6+ years in clinical operations and data quality, pursuing an M.S. in Analytics at Georgia Tech. My background bridges hands-on clinical experience — working with Epic, LIS data, and laboratory workflows at Quest Diagnostics — with graduate-level training in machine learning, NLP, and statistical modeling.

My project work focuses on applying predictive modeling, clinical NLP, and fairness-aware analysis to real-world healthcare data. I've built and tuned classification pipelines, fine-tuned transformer-based models on clinical text, and evaluated model behavior with an emphasis on interpretability, bias detection, and practical impact.

**Education:** M.S. Analytics (Computational Data Analytics track), Georgia Institute of Technology (2025–2026) — focusing on NLP, deep learning, and AI | B.S. Human Biology, Michigan State University (2018)

**Links:** [GitHub](https://github.com/cody-maddox) | [LinkedIn](https://linkedin.com/in/cody-maddox) | codymaddox616@gmail.com

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
