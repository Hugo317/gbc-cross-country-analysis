// When the search text in a dropdown changes, scroll the options list back to the top so the first matches are visible.
document.addEventListener("input", function (e) {
  if (e.target && e.target.classList && e.target.classList.contains("dash-dropdown-search")) {
    var list = e.target.closest(".dash-dropdown-content");
    var opts = list && list.querySelector(".dash-dropdown-options");
    if (opts) { opts.scrollTop = 0; }
    if (list) { list.scrollTop = 0; }
  }
}, true);
