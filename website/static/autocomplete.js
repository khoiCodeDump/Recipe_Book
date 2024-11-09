window.addedTags = new Set();
window.addedIngredients = new Set();

let addedTags = window.addedTags;
let addedIngredients = window.addedIngredients;

function autocomplete(inp, tags_inp, ingredients_inp) {
  var currentFocus;
  var debounceTimer;

  // Preprocess the data
  const allItems = [];
  if (tags_inp) {
    allItems.push(...Object.values(tags_inp).map(item => ({ ...item, type: 'Tag' })));
  }
  if (ingredients_inp) {
    allItems.push(...Object.values(ingredients_inp).map(item => ({ ...item, type: 'Ingredient' })));
  }

  function debounce(func, delay) {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(func, delay);
  }

  inp.addEventListener("input", function(e) {
    debounce(() => {
      var a, b, i, val = this.value;
      closeAllLists();
      if (!val) { return false; }
      currentFocus = -1;
      
      let parts = val.split(',');
      let lastInput = parts[parts.length - 1].trim().toLowerCase();
      
      if (lastInput.length > 0) {
        a = document.createElement("DIV");
        a.setAttribute("id", this.id + "autocomplete-list");
        a.setAttribute("class", "autocomplete-items");
        this.parentNode.appendChild(a);

        // Filter and sort matches
        const matches = allItems
          .filter(item => item.name.toLowerCase().includes(lastInput))
          .sort((a, b) => {
              const aIsTag = a.type === 'Tag' ? 0 : 1; // Assuming 'type' indicates if it's a tag or ingredient
              const bIsTag = b.type === 'Tag' ? 0 : 1;
              return aIsTag - bIsTag || a.name.toLowerCase().indexOf(lastInput) - b.name.toLowerCase().indexOf(lastInput);
          });

        matches.forEach(item => {
          b = document.createElement("DIV");
          b.innerHTML = `${item.type}: ${highlightMatch(item.name, lastInput)}`;
          b.innerHTML += `<input type='hidden' value='${item.name}'>`;
          b.addEventListener("click", function(e) {
            const selectedValue = this.getElementsByTagName("input")[0].value;
            if (addItemComponent(selectedValue, item.type, inp)) {
              // Clear input until a comma is met, or clear whole search if no comma
              let currentValue = inp.value;
              let lastCommaIndex = currentValue.lastIndexOf(',');
              if (lastCommaIndex !== -1) {
                inp.value = currentValue.substring(0, lastCommaIndex + 1).trim() + ' ';
              } else {
                inp.value = '';
              }
              closeAllLists();
              inp.focus();
            }
          });
          a.appendChild(b);
        });
      }
    }, 300);  // 300ms debounce
  });

  inp.addEventListener("keydown", function(e) {
    var x = document.getElementById(this.id + "autocomplete-list");
    if (x) x = x.getElementsByTagName("div");
    if (e.keyCode == 40) {
      currentFocus++;
      addActive(x);
    } else if (e.keyCode == 38) {
      currentFocus--;
      addActive(x);
    } else if (e.keyCode == 13) {
      e.preventDefault();
      if (currentFocus > -1 && x) {
        x[currentFocus].click();
        currentFocus = -1; // Reset currentFocus after selection
      } else if (!window.location.pathname.includes('/post_recipe')){
        console.log("Submitting form: In autocomplete");
        updateHiddenFields();
        this.form.submit();
      }
    }
  });

  function addActive(x) {
    if (!x) return false;
    removeActive(x);
    if (currentFocus >= x.length) currentFocus = 0;
    if (currentFocus < 0) currentFocus = (x.length - 1);
    x[currentFocus].classList.add("autocomplete-active");
  }

  function removeActive(x) {
    for (var i = 0; i < x.length; i++) {
      x[i].classList.remove("autocomplete-active");
    }
  }

  function closeAllLists(elmnt) {
    var x = document.getElementsByClassName("autocomplete-items");
    for (var i = 0; i < x.length; i++) {
      if (elmnt != x[i] && elmnt != inp) {
        x[i].parentNode.removeChild(x[i]);
      }
    }
  }

  document.addEventListener("click", function (e) {
    closeAllLists(e.target);
  });
}

function highlightMatch(text, query) {
  const index = text.toLowerCase().indexOf(query.toLowerCase());
  if (index >= 0) {
    return text.substring(0, index) +
           "<strong>" + text.substring(index, index + query.length) + "</strong>" +
           text.substring(index + query.length);
  }
  return text;
}

function addItemComponent(value, type, inputElement) {
  const itemSet = type === "Tag" ? addedTags : addedIngredients;
  
  if (itemSet.has(value)) {    
    // Clear the input up to the last comma for both tags and ingredients
    let currentValue = inputElement.value;
    let lastCommaIndex = currentValue.lastIndexOf(',');
    if (lastCommaIndex !== -1) {
      inputElement.value = currentValue.substring(0, lastCommaIndex + 1).trim() + ' ';
    } else {
      inputElement.value = '';
    }
    
    return false;
  }

  itemSet.add(value);

  const itemContainer = document.createElement("div");
  itemContainer.className = "item-component";
  itemContainer.setAttribute('data-type', type);
  itemContainer.innerHTML = `
    <span>${type}: ${value}</span>
    <button class="remove-item">×</button>
  `;
  
  itemContainer.querySelector('.remove-item').addEventListener('click', function() {
    itemContainer.remove();
    itemSet.delete(value);
  });

  inputElement.parentNode.insertBefore(itemContainer, inputElement.nextSibling);
  return true;
}

function updateHiddenFields() {
  const tagsInput = document.createElement('input');
  tagsInput.type = 'hidden';
  tagsInput.name = 'Tags';
  tagsInput.value = Array.from(addedTags).join(',');

  const ingredientsInput = document.createElement('input');
  ingredientsInput.type = 'hidden';
  ingredientsInput.name = 'Ingredients';
  ingredientsInput.value = Array.from(addedIngredients).join(',');

  const form = document.querySelector('form');
  form.appendChild(tagsInput);
  form.appendChild(ingredientsInput);
}