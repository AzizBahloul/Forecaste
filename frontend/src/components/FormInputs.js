import React from 'react';

const FormInputs = ({ formData, handleChange }) => {
  return (
    <div>
      <div className="form-group">
        <label htmlFor="new_base_price">New Base Price</label>
        <input
          id="new_base_price"
          name="new_base_price"
          type="number"
          className="input-field"
          value={formData.new_base_price}
          onChange={handleChange}
          placeholder="Enter new base price..."
          required
        />
      </div>

      <div className="form-group">
        <label htmlFor="new_total_price">New Total Price</label>
        <input
          id="new_total_price"
          name="new_total_price"
          type="number"
          className="input-field"
          value={formData.new_total_price}
          onChange={handleChange}
          placeholder="Enter new total price..."
          required
        />
      </div>

      <div className="form-group">
        <label htmlFor="new_is_featured_sku">Is Featured SKU</label>
        <select
          id="new_is_featured_sku"
          name="new_is_featured_sku"
          className="input-field"
          value={formData.new_is_featured_sku ? "yes" : "no"}
          onChange={(e) =>
            handleChange({
              target: {
                name: e.target.name,
                value: e.target.value === "yes",
              },
            })
          }
          required
        >
          <option value="">Select...</option>
          <option value="yes">Yes</option>
          <option value="no">No</option>
        </select>
      </div>

      <div className="form-group">
        <label htmlFor="new_is_display_sku">Is Display SKU</label>
        <select
          id="new_is_display_sku"
          name="new_is_display_sku"
          className="input-field"
          value={formData.new_is_display_sku ? "yes" : "no"}
          onChange={(e) =>
            handleChange({
              target: {
                name: e.target.name,
                value: e.target.value === "yes",
              },
            })
          }
          required
        >
          <option value="">Select...</option>
          <option value="yes">Yes</option>
          <option value="no">No</option>
        </select>
      </div>

      <div className="form-group">
        <label htmlFor="new_sku_id">SKU ID</label>
        <input
          id="new_sku_id"
          name="new_sku_id"
          type="text"
          className="input-field"
          value={formData.new_sku_id}
          onChange={handleChange}
          placeholder="Enter new SKU ID..."
          required
        />
      </div>
    </div>
  );
};

export default FormInputs;