{{ name | escape | underline}}

.. automodule:: {{ fullname }}
   :members:

{% block modules %}
{% if modules %}

.. rubric:: Submodules

.. autosummary::
   :toctree:
   :template: module.rst
   :recursive:
   {% for item in modules %}
      {{ item }}
   {%- endfor %}

{% endif %}
{% endblock %}
